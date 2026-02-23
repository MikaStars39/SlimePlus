import asyncio
import logging

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking

from slime_plus.infer import run_streaming_inference

logger = logging.getLogger(__name__)

def add_plus_arguments(parser):
    parser.add_argument(
        "--plus-input-path",
        type=str,
        required=True,
        help="Input jsonl path for plus streaming inference.",
    )
    parser.add_argument(
        "--plus-output-path",
        type=str,
        default=None,
        help="Output jsonl path for plus streaming inference.",
    )
    parser.add_argument(
        "--plus-num-workers",
        type=int,
        default=1,
        help="Number of Ray AsyncRolloutWorker actors.",
    )
    parser.add_argument(
        "--plus-flush-every",
        type=int,
        default=32,
        help="Flush interval (records) for sink fsync.",
    )
    parser.add_argument(
        "--plus-worker-concurrency",
        type=int,
        default=10,
        help="Per-worker async consumer count.",
    )
    parser.add_argument(
        "--plus-worker-batch-size",
        type=int,
        default=None,
        help="Per-worker producer batch size. Defaults to rollout_batch_size.",
    )
    parser.add_argument(
        "--plus-sink-flush-size",
        type=int,
        default=32,
        help="Batch size sent from worker to sink actor.",
    )
    return parser


def train(args):
    configure_logger()
    init_tracking(args)

    rollout_manager = None
    if args.sglang_router_ip is None:
        # RolloutManager bootstrapping still constructs the default rollout data source,
        # which requires prompt_data to be a valid path.
        if getattr(args, "prompt_data", None) is None:
            args.prompt_data = args.plus_input_path
            logger.info("prompt_data is None, fallback to plus_input_path for rollout bootstrap: %s", args.prompt_data)

        # If router is not provided, bootstrap rollout engines and router as before.
        pgs = create_placement_groups(args)
        rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])
        logger.info("Waiting for Rollout Engines and Router to initialize...")
        ray.get(rollout_manager.get_rollout_engines_and_lock.remote())
        logger.info("All engines are initialized and weights are loaded.")
    else:
        logger.info(
            "Using existing router at %s:%s",
            args.sglang_router_ip,
            args.sglang_router_port,
        )
    
    asyncio.run(run_streaming_inference(args))

    if rollout_manager is not None:
        ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_plus_arguments)
    train(args)
