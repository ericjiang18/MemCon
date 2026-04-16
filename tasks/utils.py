def get_model_type(model_name: str) -> str:
    
    valid_model_types: list[str] = [
        'gpt-4o-mini',
        'gpt-4.1-mini',
        'gpt-4.1-nano',
        'gpt-5.4-nano',
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
        'qwen2.5-7b', 
        'qwen2.5-14b',
        'qwen2.5-32b', 
        'qwen2.5-72b',
        'qwen3-flash',
        'qwen3-max',
        'intern', 
        'deepseek-v3',
    ]

    for model_type in valid_model_types:
        if model_type in model_name.lower():
            return model_type
    
    if model_name.lower().startswith('gemini'):
        return model_name.lower()
    
    return 'unknown'