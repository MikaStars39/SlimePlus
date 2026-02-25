import asyncio
import logging
import time
import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.http_utils import _wrap_ipv6, find_available_port, get_host_info
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
    parser.add_argument(
        "--plus-progress-interval-sec",
        type=float,
        default=100.0,
        help="Progress/TPS logging interval in seconds.",
    )
    parser.add_argument(
        "--plus-server-only",
        action="store_true",
        default=False,
        help="Only start/keep rollout service; do not run streaming inference.",
    )
    parser.add_argument(
        "--plus-router-addr",
        type=str,
        default=None,
        help="Existing router endpoint in ip:port format. If set, skip local service startup and send requests to this router.",
    )
    return parser


def train(args):
    configure_logger()
    init_tracking(args)

    if args.plus_router_addr:
        endpoint = args.plus_router_addr.strip()
        if ":" not in endpoint:
            raise ValueError(f"Invalid --plus-router-addr: {endpoint}. Expected format ip:port")
        host, port_str = endpoint.rsplit(":", 1)
        host = host.strip()
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if not host:
            raise ValueError(f"Invalid --plus-router-addr host: {endpoint}")
        try:
            port = int(port_str)
        except ValueError as exc:
            raise ValueError(f"Invalid --plus-router-addr port: {endpoint}") from exc
        args.sglang_router_ip = _wrap_ipv6(host)
        args.sglang_router_port = port
        logger.info("Using router from --plus-router-addr: %s:%s", args.sglang_router_ip, args.sglang_router_port)

    rollout_manager = None
    try:
        if args.sglang_router_ip is None:
            # Let RolloutManager start router in its own _start_router path.
            # We only pin a deterministic port so driver-side plus workers can use the same endpoint later.
            if args.sglang_router_port is None:
                args.sglang_router_port = find_available_port(3500)

            # RolloutManager bootstrapping still constructs the default rollout data source,
            # which requires prompt_data to be a valid path.
            if getattr(args, "prompt_data", None) is None:
                args.prompt_data = args.plus_input_path
                logger.info("prompt_data fallback to plus_input_path: %s", args.prompt_data)

            pgs = create_placement_groups(args)
            rollout_manager, _ = create_rollout_manager(args, pgs["rollout"])
            logger.info("Waiting for Rollout Engines and Router to initialize...")
            ray.get(rollout_manager.get_rollout_engines_and_lock.remote())
            logger.info("All engines are initialized and weights are loaded.")

            # RolloutManager mutates its own args copy, not the driver's.
            # Backfill driver args so plus workers can resolve router endpoint.
            args.sglang_router_ip = _wrap_ipv6(get_host_info()[1])
            logger.info("Using router endpoint %s:%s", args.sglang_router_ip, args.sglang_router_port)
        else:
            logger.info(
                "Using existing router at %s:%s",
                args.sglang_router_ip,
                args.sglang_router_port,
            )

        if args.plus_server_only:
            logger.info(
                "plus_server_only enabled. Service is ready at router %s:%s. Waiting for external requests (Ctrl+C to stop).",
                args.sglang_router_ip,
                args.sglang_router_port,
            )
            while True:
                time.sleep(60)
        else:
            asyncio.run(run_streaming_inference(args))
    finally:
        if rollout_manager is not None:
            ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_plus_arguments)
    train(args)
