"""
Image processing utilities for the visual search engine.
Handles validation, preprocessing, and optimization.
"""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
import logging
import hashlib
import io

from ..config import Config
from .logger import get_logger

logger = get_logger(__name__)

def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate if an image file is readable and in supported format.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            return False
        
        # Check file extension
        if image_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
            return False
        
        # Try to open and validate image
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
            
        # Double-check by opening again (verify() corrupts the image object)
        with Image.open(image_path) as img:
            img.load()  # Actually load the image data
            
        return True
        
    except Exception as e:
        logger.debug(f"Image validation failed for {image_path}: {str(e)}")
        return False

def get_image_info(image_path: Union[str, Path]) -> dict:
    """
    Get detailed information about an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    try:
        image_path = Path(image_path)
        
        with Image.open(image_path) as img:
            info = {
                'filename': image_path.name,
                'filepath': str(image_path),
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'file_size_bytes': image_path.stat().st_size,
                'has_transparency': 'transparency' in img.info or img.mode in ('RGBA', 'LA'),
            }
            
            # Add EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                info['has_exif'] = True
                exif = img._getexif()
                if exif:
                    # Add common EXIF tags
                    info['exif'] = {
                        'orientation': exif.get(274),  # Orientation tag
                        'datetime': exif.get(306),     # DateTime tag
                        'camera_make': exif.get(271),  # Make tag
                        'camera_model': exif.get(272), # Model tag
                    }
            else:
                info['has_exif'] = False
            
            return info
            
    except Exception as e:
        logger.error(f"Failed to get image info for {image_path}: {str(e)}")
        return {}

def resize_image(
    image: Image.Image, 
    target_size: Union[int, Tuple[int, int]], 
    maintain_aspect_ratio: bool = True,
    resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image with optional aspect ratio preservation.
    
    Args:
        image: PIL Image to resize
        target_size: Target size (int for square, tuple for (width, height))
        maintain_aspect_ratio: Whether to maintain aspect ratio
        resample: Resampling algorithm
        
    Returns:
        Resized PIL Image
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    if maintain_aspect_ratio:
        # Calculate size maintaining aspect ratio
        image.thumbnail(target_size, resample)
        
        # Pad to exact target size if needed
        if image.size != target_size:
            # Create new image with target size and paste resized image centered
            new_image = Image.new(image.mode, target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.width) // 2
            paste_y = (target_size[1] - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            return new_image
        
        return image
    else:
        # Direct resize without maintaining aspect ratio
        return image.resize(target_size, resample)

def normalize_image(image: Image.Image) -> Image.Image:
    """
    Normalize image for consistent processing.
    
    Args:
        image: PIL Image to normalize
        
    Returns:
        Normalized PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Auto-orient based on EXIF data
    image = ImageOps.exif_transpose(image)
    
    return image

def create_image_hash(image: Image.Image) -> str:
    """
    Create a hash of an image for deduplication.
    
    Args:
        image: PIL Image
        
    Returns:
        MD5 hash string
    """
    # Resize to small size for consistent hashing
    small_image = image.resize((8, 8), Image.Resampling.LANCZOS)
    small_image = small_image.convert('L')  # Grayscale
    
    # Get image bytes
    img_bytes = io.BytesIO()
    small_image.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    
    # Create hash
    return hashlib.md5(img_data).hexdigest()

def detect_duplicates(image_paths: List[Union[str, Path]]) -> List[List[str]]:
    """
    Detect duplicate images based on content hashing.
    
    Args:
        image_paths: List of image paths
        
    Returns:
        List of duplicate groups (each group contains paths of duplicate images)
    """
    hash_to_paths = {}
    
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img_hash = create_image_hash(img)
                
                if img_hash not in hash_to_paths:
                    hash_to_paths[img_hash] = []
                hash_to_paths[img_hash].append(str(image_path))
                
        except Exception as e:
            logger.warning(f"Failed to process {image_path} for duplicate detection: {e}")
            continue
    
    # Return only groups with duplicates
    duplicates = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    
    if duplicates:
        logger.info(f"Found {len(duplicates)} groups of duplicate images")
    
    return duplicates

def crop_to_square(image: Image.Image, crop_position: str = 'center') -> Image.Image:
    """
    Crop image to square aspect ratio.
    
    Args:
        image: PIL Image to crop
        crop_position: Where to crop from ('center', 'top', 'bottom')
        
    Returns:
        Square-cropped PIL Image
    """
    width, height = image.size
    
    if width == height:
        return image
    
    # Determine crop size (smaller dimension)
    crop_size = min(width, height)
    
    if crop_position == 'center':
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
    elif crop_position == 'top':
        left = (width - crop_size) // 2
        top = 0
    elif crop_position == 'bottom':
        left = (width - crop_size) // 2
        top = height - crop_size
    else:
        # Default to center
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
    
    right = left + crop_size
    bottom = top + crop_size
    
    return image.crop((left, top, right, bottom))

def augment_image(image: Image.Image, augmentation_type: str = 'light') -> Image.Image:
    """
    Apply augmentation to image for data diversity.
    
    Args:
        image: PIL Image to augment
        augmentation_type: Type of augmentation ('light', 'medium', 'heavy')
        
    Returns:
        Augmented PIL Image
    """
    import random
    from PIL import ImageEnhance, ImageFilter
    
    augmented = image.copy()
    
    if augmentation_type in ['light', 'medium', 'heavy']:
        # Random brightness adjustment
        enhancer = ImageEnhance.Brightness(augmented)
        factor = random.uniform(0.9, 1.1) if augmentation_type == 'light' else random.uniform(0.8, 1.2)
        augmented = enhancer.enhance(factor)
        
        # Random contrast adjustment
        enhancer = ImageEnhance.Contrast(augmented)
        factor = random.uniform(0.9, 1.1) if augmentation_type == 'light' else random.uniform(0.8, 1.2)
        augmented = enhancer.enhance(factor)
    
    if augmentation_type in ['medium', 'heavy']:
        # Random saturation adjustment
        enhancer = ImageEnhance.Color(augmented)
        factor = random.uniform(0.8, 1.2)
        augmented = enhancer.enhance(factor)
        
        # Small rotation
        angle = random.uniform(-5, 5) if augmentation_type == 'medium' else random.uniform(-10, 10)
        augmented = augmented.rotate(angle, fillcolor=(255, 255, 255))
    
    if augmentation_type == 'heavy':
        # Slight blur or sharpening
        if random.choice([True, False]):
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))
        else:
            augmented = augmented.filter(ImageFilter.UnsharpMask())
    
    return augmented

def batch_process_images(
    image_paths: List[Union[str, Path]],
    output_dir: Path,
    target_size: Union[int, Tuple[int, int]] = 224,  # Default CLIP image size
    quality: int = 95
) -> List[Path]:
    """
    Batch process images with resizing and optimization.
    
    Args:
        image_paths: List of input image paths
        output_dir: Directory to save processed images
        target_size: Target size for resizing
        quality: JPEG quality for compression
        
    Returns:
        List of processed image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    processed_paths = []
    failed_count = 0
    
    for image_path in image_paths:
        try:
            image_path = Path(image_path)
            
            with Image.open(image_path) as img:
                # Normalize and resize
                img = normalize_image(img)
                img = resize_image(img, target_size)
                
                # Create output path
                output_path = output_dir / f"{image_path.stem}_processed{image_path.suffix}"
                
                # Save with optimization
                save_kwargs = {'optimize': True}
                if img.format in ['JPEG', 'JPG'] or output_path.suffix.lower() in ['.jpg', '.jpeg']:
                    save_kwargs['quality'] = quality
                
                img.save(output_path, **save_kwargs)
                processed_paths.append(output_path)
                
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            failed_count += 1
            continue
    
    logger.info(f"Batch processing completed: {len(processed_paths)} success, {failed_count} failed")
    return processed_paths

def estimate_memory_usage(image_size: Tuple[int, int], batch_size: int) -> dict:
    """
    Estimate memory usage for batch processing.
    
    Args:
        image_size: Size of images (width, height)
        batch_size: Number of images in batch
        
    Returns:
        Memory usage estimates in MB
    """
    width, height = image_size
    channels = 3  # RGB
    
    # Raw image data (float32)
    raw_mb_per_image = (width * height * channels * 4) / (1024 * 1024)
    
    # Tensor overhead (approximately 2x for PyTorch tensors)
    tensor_mb_per_image = raw_mb_per_image * 2
    
    # Batch memory usage
    batch_memory_mb = tensor_mb_per_image * batch_size
    
    # Additional overhead for model processing (approximately 1.5x)
    total_memory_mb = batch_memory_mb * 1.5
    
    return {
        'raw_mb_per_image': raw_mb_per_image,
        'tensor_mb_per_image': tensor_mb_per_image,
        'batch_memory_mb': batch_memory_mb,
        'total_memory_mb': total_memory_mb,
        'recommended_max_batch_size': int(Config.MAX_MEMORY_GB * 1024 / tensor_mb_per_image)
    }