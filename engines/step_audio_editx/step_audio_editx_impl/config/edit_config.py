"""
音频编辑配置模块
包含支持的编辑类型和相关配置
"""

def get_supported_edit_types():
    """
    获取支持的编辑类型和选项

    Returns:
        Dict[str, list]: Dictionary of edit types and their options
    """
    return {
        "clone": [],
        "emotion": [
            'happy', 'angry', 'sad', 'humour', 'confusion', 'disgusted',
            'empathy', 'embarrass', 'fear', 'surprised', 'excited',
            'depressed', 'coldness', 'admiration', 'remove'
        ],
        "style": [
            'serious', 'arrogant', 'child', 'older', 'girl', 'pure',
            'sister', 'sweet', 'ethereal', 'whisper', 'gentle', 'recite',
            'generous', 'act_coy', 'warm', 'shy', 'comfort', 'authority',
            'chat', 'radio', 'soulful', 'story', 'vivid', 'program',
            'news', 'advertising', 'roar', 'murmur', 'shout', 'deeply', 'loudly',
            'remove', 'exaggerated'
        ],
        "vad": [],
        "denoise": [],
        "paralinguistic": [],
        "speed": ["faster", "slower", "more faster", "more slower"],
    }