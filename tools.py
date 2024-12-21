tools = [
        {
            "type": "function",
            "function": {
                "name": "search_db",
                "description": "Get the data using vector search. Provide the name of the collection and input_query to perform vector search in the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to perform vector search in.",
                        },
                        "input_query": {
                            "type": "string",
                            "description": "text to convert to embeddings and perform vector search in the database",
                        },
                        "n": {
                            "type": "integer",
                            "description": "number of results to retrive from the search (Top K)",
                        },
                    },
                    "required": ["collection_name", "input_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Internet_search",
                "description": "Tool to perform internet search using duckduckgo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Query to use while searching the internet"},
                    },
                    "required": ["input"],
                },
            },
        }
    ]