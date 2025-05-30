def get_message_template(prefix, suffix, lang="python"):

    message_templates = {
       "default": [
            {
                "role": "user",
                "content": f"Please complete the code in the middle given the prefix code snippet and suffix code snippet. Please respond in the markdown format: ```\n<code in the middle>\n```\n\nSuffix:{suffix}\nPrefix: {prefix}\nCode in the middle:"
            }
        ]
        
    }
    

    return message_templates['default']
