import sys
from analyzer.ast_parser import parse_and_detect
from analyzer.feature_builder import build_feature_dict
from collector.sys_state import get_system_state
from model.inference import predict  
from collector.device_runner import run_on_device
from utils.logger import get_logger

logger = get_logger("main")

def run_pipeline(source_file):
    # Static analysis 
    analysis = parse_and_detect(source_file)
    op_type = analysis['operation_type']
    log_size = analysis['log_total_sizes']   

    # Dynamic system state
    sys_state = get_system_state()

    # Feature vector
    features = build_feature_dict(op_type, log_size, sys_state)
    logger.info("Final feature dict sent to model: %s", features)

    # Model prediction
    decision = predict(features)  
    logger.info("Prediction: %s", decision)

    # Execute on predicted device
    run_on_device(source_file, decision, extra_info=analysis)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    run_pipeline(sys.argv[1])
