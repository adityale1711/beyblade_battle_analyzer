import torch
from src.beyblade_battle_analyzer import logger


class DeviceManager:
    """
    A utility class for managing device selection for model training and inference.
    """
    
    @staticmethod
    def get_device(device_config: str = "auto") -> str:
        """
        Determines the best device to use based on configuration and availability.
        
        :param device_config: Device configuration string. Options:
                            - 'auto': Automatically select the best available device
                            - 'cpu': Force CPU usage
                            - 'cuda': Use CUDA if available, fallback to CPU
                            - 'cuda:0', 'cuda:1', etc.: Use specific GPU
        :return: Device string suitable for PyTorch/YOLO
        """
        
        # If explicitly requesting CPU
        if device_config.lower() == 'cpu':
            logger.info("Device set to CPU (explicitly requested)")
            return 'cpu'
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        if device_config.lower() == 'auto':
            if cuda_available:
                device = f'cuda:{torch.cuda.current_device()}'
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
                logger.info(f"Device auto-selected: {device} ({gpu_name}) - {gpu_count} GPU(s) available")
                return device
            else:
                logger.info("Device auto-selected: CPU (CUDA not available)")
                return 'cpu'
        
        # Handle specific CUDA device requests
        if device_config.lower().startswith('cuda'):
            if not cuda_available:
                logger.warning(f"CUDA device '{device_config}' requested but CUDA is not available. Falling back to CPU.")
                return 'cpu'
            
            # Extract device number if specified
            if ':' in device_config:
                try:
                    device_num = int(device_config.split(':')[1])
                    if device_num >= torch.cuda.device_count():
                        logger.warning(f"CUDA device {device_num} requested but only {torch.cuda.device_count()} GPU(s) available. Using GPU 0.")
                        device = 'cuda:0'
                    else:
                        device = device_config
                except ValueError:
                    logger.warning(f"Invalid CUDA device format '{device_config}'. Using default CUDA device.")
                    device = 'cuda:0'
            else:
                device = 'cuda:0'
            
            gpu_name = torch.cuda.get_device_name(device)
            logger.info(f"Device set to: {device} ({gpu_name})")
            return device
        
        # Fallback for unrecognized device config
        logger.warning(f"Unrecognized device configuration '{device_config}'. Falling back to auto-selection.")
        return DeviceManager.get_device('auto')
    
    @staticmethod
    def get_device_info() -> dict:
        """
        Get comprehensive information about available devices.
        
        :return: Dictionary containing device information
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'gpu_names': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_total_gb = memory_total / (1024**3)
                info['gpu_names'].append({
                    'index': i,
                    'name': gpu_name,
                    'memory_gb': round(memory_total_gb, 2)
                })
        
        return info
    
    @staticmethod
    def log_device_info():
        """
        Log comprehensive device information.
        """
        info = DeviceManager.get_device_info()
        
        logger.info("=== Device Information ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {info['cuda_available']}")
        
        if info['cuda_available']:
            logger.info(f"CUDA version: {info['cuda_version']}")
            logger.info(f"GPU count: {info['gpu_count']}")
            logger.info(f"Current device: {info['current_device']}")
            
            for gpu in info['gpu_names']:
                logger.info(f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']} GB)")
        else:
            logger.info("No CUDA devices available")
        
        logger.info("========================")
    
    @staticmethod
    def validate_device_config(device_config: str) -> bool:
        """
        Validate if the device configuration is valid.
        
        :param device_config: Device configuration string
        :return: True if valid, False otherwise
        """
        valid_configs = ['auto', 'cpu', 'cuda']
        
        if device_config.lower() in valid_configs:
            return True
        
        # Check CUDA device format (cuda:X)
        if device_config.lower().startswith('cuda:'):
            try:
                device_num = int(device_config.split(':')[1])
                return device_num >= 0
            except (ValueError, IndexError):
                return False
        
        return False
