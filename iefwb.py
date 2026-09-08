list1 = [0,1,2,3,4,5,6,7,8]
print(list1[:-3])


"Extract parameters in JSON format.\n\n"
"Example 1:\n"
"Function: fn_add_numbers\n"
"User Request: Add 5 and 3\n"
"Function call: fn_add_numbers{\"a\":5,\"b\":3}\n\n"
"Example 2:\n"
"Function: fn_substitute_string_with_regex\n"
"User Request: Replace digits in 'Hello 123' with 'X'\n"
"Function call: fn_substitute_string_with_regex{\"source_string\":\"Hello 123\",\"regex\":\"[0-9]+\",\"replacement\":\"X\"}\n\n"
f"Function: {chosen_def.name}\n"
f"Description: {chosen_def.description}\n"
f"Parameters: {', '.join(f'{k}: {v.type.name}' for k, v in chosen_def.parameters.items())}\n"
f"User Request: {prompt}\n"
"Function call: "

