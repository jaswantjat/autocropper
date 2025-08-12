"""
Configuration system for Card Rectification Flask API
Handles environment variables with sensible defaults
"""

import os
import logging
from pathlib import Path
from typing import List, Optional


class Config:
    """Base configuration class with environment variable support."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self._load_config()
    
    def _load_config(self):
        """Load all configuration from environment variables."""
        # Basic Flask settings
        self.DEBUG = self._get_bool('DEBUG', False)
        self.TESTING = self._get_bool('TESTING', False)
        self.HOST = self._get_str('HOST', '0.0.0.0')
        self.PORT = self._get_int('PORT', 5000)
        
        # Security settings
        self.SECRET_KEY = self._get_str('SECRET_KEY', 'dev-secret-key-change-in-production')
        self.MAX_CONTENT_LENGTH = self._get_int('MAX_CONTENT_LENGTH_MB', 16) * 1024 * 1024
        self.ALLOWED_EXTENSIONS = self._get_list('ALLOWED_EXTENSIONS', ['png', 'jpg', 'jpeg', 'bmp'])
        
        # File and directory settings
        self.BASE_DIR = Path(__file__).parent.absolute()
        self.UPLOAD_FOLDER = self._get_path('UPLOAD_FOLDER', self.BASE_DIR / 'uploads')
        self.TEMP_FOLDER = self._get_path('TEMP_FOLDER', self.BASE_DIR / 'temp')
        self.RESULTS_FOLDER = self._get_path('RESULTS_FOLDER', self.BASE_DIR / 'results')
        
        # Model settings
        self.MODEL_PATH = self._get_path('MODEL_PATH', self.BASE_DIR / 'CRDN1000.pkl')
        self.MODEL_ARCH = self._get_str('MODEL_ARCH', 'UNetRNN')
        self.FORCE_CPU = self._get_bool('FORCE_CPU', False)
        self.MODEL_TIMEOUT = self._get_int('MODEL_TIMEOUT', 30)
        
        # Processing parameters
        self.PROCESS_SIZE = self._get_int('PROCESS_SIZE', 1000)
        self.MODEL_INPUT_SIZE = self._get_int('MODEL_INPUT_SIZE', 1000)
        self.PROCESSING_TIMEOUT = self._get_int('PROCESSING_TIMEOUT', 120)
        self.MAX_IMAGE_DIMENSION = self._get_int('MAX_IMAGE_DIMENSION', 5000)
        self.MIN_IMAGE_DIMENSION = self._get_int('MIN_IMAGE_DIMENSION', 100)
        
        # Logging settings
        self.LOG_LEVEL = self._get_str('LOG_LEVEL', 'INFO')
        self.LOG_FILE = self._get_str('LOG_FILE', None)
        self.LOG_FORMAT = self._get_str('LOG_FORMAT', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Performance settings
        self.WORKERS = self._get_int('WORKERS', 1)
        self.WORKER_TIMEOUT = self._get_int('WORKER_TIMEOUT', 120)
        self.KEEP_ALIVE = self._get_int('KEEP_ALIVE', 2)
        self.MAX_REQUESTS = self._get_int('MAX_REQUESTS', 1000)
        self.MAX_REQUESTS_JITTER = self._get_int('MAX_REQUESTS_JITTER', 50)
        
        # Cleanup settings
        self.AUTO_CLEANUP = self._get_bool('AUTO_CLEANUP', True)
        self.CLEANUP_INTERVAL = self._get_int('CLEANUP_INTERVAL', 3600)  # 1 hour
        self.TEMP_FILE_MAX_AGE = self._get_int('TEMP_FILE_MAX_AGE', 1800)  # 30 minutes
        
        # API settings
        self.API_TITLE = self._get_str('API_TITLE', 'Card Rectification API')
        self.API_VERSION = self._get_str('API_VERSION', '2.0.0')
        self.API_DESCRIPTION = self._get_str('API_DESCRIPTION', 
            'REST API for ID card rectification using deep learning')
        
        # Development settings
        self.ENABLE_CORS = self._get_bool('ENABLE_CORS', self.DEBUG)
        self.SAVE_DEBUG_IMAGES = self._get_bool('SAVE_DEBUG_IMAGES', self.DEBUG)
        self.DEBUG_IMAGE_FOLDER = self._get_path('DEBUG_IMAGE_FOLDER', self.BASE_DIR / 'debug')
        
        # Validation
        self._validate_config()
        
        # Create necessary directories
        self._create_directories()
    
    def _get_str(self, key: str, default: str) -> str:
        """Get string value from environment."""
        return os.getenv(key, default)
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer value from environment."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def _get_list(self, key: str, default: List[str]) -> List[str]:
        """Get list value from environment (comma-separated)."""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(',')]
        return default
    
    def _get_path(self, key: str, default: Path) -> Path:
        """Get path value from environment."""
        value = os.getenv(key)
        if value:
            return Path(value).absolute()
        return default
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate port range
        if not (1 <= self.PORT <= 65535):
            raise ValueError(f"Invalid port number: {self.PORT}")
        
        # Validate file size limits
        if self.MAX_CONTENT_LENGTH <= 0:
            raise ValueError(f"Invalid max content length: {self.MAX_CONTENT_LENGTH}")
        
        # Validate image dimensions
        if self.MIN_IMAGE_DIMENSION >= self.MAX_IMAGE_DIMENSION:
            raise ValueError("MIN_IMAGE_DIMENSION must be less than MAX_IMAGE_DIMENSION")
        
        # Validate processing sizes
        if self.PROCESS_SIZE <= 0 or self.MODEL_INPUT_SIZE <= 0:
            raise ValueError("Processing sizes must be positive")
        
        # Validate timeouts
        if self.PROCESSING_TIMEOUT <= 0 or self.MODEL_TIMEOUT <= 0:
            raise ValueError("Timeouts must be positive")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.LOG_LEVEL}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.UPLOAD_FOLDER,
            self.TEMP_FOLDER,
            self.RESULTS_FOLDER
        ]
        
        if self.SAVE_DEBUG_IMAGES:
            directories.append(self.DEBUG_IMAGE_FOLDER)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_log_level(self) -> int:
        """Get numeric log level for logging configuration."""
        return getattr(logging, self.LOG_LEVEL.upper())
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG and not self.TESTING
    
    def get_model_path_str(self) -> str:
        """Get model path as string."""
        return str(self.MODEL_PATH)
    
    def get_upload_folder_str(self) -> str:
        """Get upload folder as string."""
        return str(self.UPLOAD_FOLDER)
    
    def get_temp_folder_str(self) -> str:
        """Get temp folder as string."""
        return str(self.TEMP_FOLDER)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for debugging."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict
    
    def print_config(self):
        """Print current configuration (for debugging)."""
        print("Current Configuration:")
        print("=" * 50)
        for key, value in self.to_dict().items():
            # Hide sensitive information
            if 'SECRET' in key.upper() or 'KEY' in key.upper():
                value = '*' * len(str(value))
            print(f"{key}: {value}")
        print("=" * 50)


class DevelopmentConfig(Config):
    """Development-specific configuration."""
    
    def __init__(self):
        super().__init__()
        # Override defaults for development
        if not os.getenv('DEBUG'):
            self.DEBUG = True
        if not os.getenv('LOG_LEVEL'):
            self.LOG_LEVEL = 'DEBUG'
        if not os.getenv('SAVE_DEBUG_IMAGES'):
            self.SAVE_DEBUG_IMAGES = True


class ProductionConfig(Config):
    """Production-specific configuration."""
    
    def __init__(self):
        super().__init__()
        # Override defaults for production
        if not os.getenv('DEBUG'):
            self.DEBUG = False
        if not os.getenv('LOG_LEVEL'):
            self.LOG_LEVEL = 'INFO'
        if not os.getenv('SAVE_DEBUG_IMAGES'):
            self.SAVE_DEBUG_IMAGES = False
        
        # Require secret key in production
        if self.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("SECRET_KEY must be set in production")


class TestingConfig(Config):
    """Testing-specific configuration."""
    
    def __init__(self):
        super().__init__()
        # Override defaults for testing
        self.TESTING = True
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'
        self.AUTO_CLEANUP = False  # Don't auto-cleanup during tests


# Configuration factory
def get_config(config_name: Optional[str] = None) -> Config:
    """
    Get configuration instance based on environment.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, uses FLASK_ENV environment variable
    
    Returns:
        Configuration instance
    """
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_map.get(config_name, DevelopmentConfig)
    return config_class()


# Global configuration instance
config = get_config()
