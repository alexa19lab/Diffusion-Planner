
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config
from pathlib import Path
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from diffusion_planner.utils.nuplan_scenario_render import NuplanScenarioRender
from diffusion_planner.utils.scenario_manager import ScenarioManager
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(AbstractPlanner):
    requires_scenario: bool = True
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            render: bool = True,
            save_dir: str = "/mnt/csi-data-aly/shared/public/ruiwan/nuplan_vis_20pth_no_interpolation/",
            scenario: AbstractScenario = None,
            eval_dt: float = 0.1,
            eval_num_frames: int = 80,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._scenario_type = scenario.scenario_type

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

        self._render = render
        self._imgs = []
        self._scenario = scenario
        if render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir)
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)


    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict:Dict = torch.load(self._ckpt_path)
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    
        predictions = outputs['prediction'][0, 0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
            if isinstance(global_trajectory, InterpolatedTrajectory):
                states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)        
        _, outputs = self._planner(inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        predictions = outputs["prediction"][0, 0, 1:].detach().cpu().numpy().astype(np.float64)
        candidate_trajectories = outputs["prediction"][0, 1:, 0].detach().cpu().numpy().astype(np.float64)
        vis_trajectory = outputs["prediction"][0, 0, 0].detach().cpu().numpy().astype(np.float64)

        if self._render:
            print("yes we are rendering")
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    planning_trajectory=vis_trajectory,
                    candidate_trajectories=candidate_trajectories,
                    # candidate_index=best_candidate_idx,
                    predictions=predictions,
                    return_img=True,
                )
            )

        return trajectory
    
    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            import imageio

            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}.mp4",
                self._imgs,
                fps=10,
            )
            print("\n video saved to ", self.video_dir / "video.mp4\n")

        return report
    