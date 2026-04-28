from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
import json
from src.common.logger import get_logger

logger = get_logger("pydantic_helper")


def _recover_schema_value(value):
    """Recover an instance-like value from common schema-shaped LLM output."""
    if isinstance(value, list):
        recovered_items = [_recover_schema_value(item) for item in value]
        return [item for item in recovered_items if item is not None]

    if not isinstance(value, dict):
        return value

    if "value" in value:
        return _recover_schema_value(value["value"])

    for variant_key in ("anyOf", "oneOf", "allOf"):
        variants = value.get(variant_key)
        if isinstance(variants, list):
            recovered_variants = [_recover_schema_value(item) for item in variants]
            recovered_variants = [item for item in recovered_variants if item is not None]
            if recovered_variants:
                return recovered_variants[0]

    if isinstance(value.get("enum"), list) and value["enum"]:
        return _recover_schema_value(value["enum"][0])

    items = value.get("items")
    if isinstance(items, list):
        return [_recover_schema_value(item) for item in items]
    if isinstance(items, dict):
        nested_items = items.get("items")
        if isinstance(nested_items, list):
            return [_recover_schema_value(item) for item in nested_items]

    if "properties" in value and isinstance(value["properties"], dict):
        return {
            key: _recover_schema_value(val)
            for key, val in value["properties"].items()
        }

    non_schema_keys = {
        key: val
        for key, val in value.items()
        if key
        not in {
            "type",
            "title",
            "description",
            "default",
            "$schema",
            "required",
            "definitions",
            "$defs",
        }
    }
    if non_schema_keys:
        return {
            key: _recover_schema_value(val)
            for key, val in non_schema_keys.items()
        }

    return None


def _normalize_schema_echo(data: dict, schema: type[BaseModel]) -> dict:
    """Normalize common LLM schema-echo output into a schema instance payload."""
    if not isinstance(data, dict):
        return data

    schema_keys = set(schema.model_fields.keys())

    candidate = data
    # Case 1: LLM returns {"properties": {...}}
    if "properties" in data and isinstance(data["properties"], dict):
        properties = data["properties"]
        # If properties contains actual data matching schema keys, use it
        if any(k in properties for k in schema_keys):
            candidate = properties
    
    # Case 2: LLM returns the schema itself (e.g. from Ollama)
    # We look for fields like "type": "string" inside properties and try to find values
    normalized = {}
    for field_name in schema.model_fields:
        if field_name in candidate:
            val = candidate[field_name]
            # If the value looks like a schema definition (dict with "type" or "items")
            # and it's NOT supposed to be a dict (or the dict doesn't match the schema)
            # then we try to recover.
            recovered = _recover_schema_value(val)
            if recovered is not None:
                normalized[field_name] = recovered
            else:
                normalized[field_name] = val

    return normalized or candidate


def _coerce_message_text(value) -> str:
    """Coerce LangChain content variants into plain text for JSON parsing."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in value
        )
    return str(value)

def build_structured_prompt(prompt: str, schema: type[BaseModel]) -> str:
    """
    Append JSON schema instructions to a prompt so the LLM understands what
    structure to output.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)

    return f"""
{prompt}

CRITICAL OUTPUT INSTRUCTIONS:
1. YOU MUST OUTPUT A COMPLETED JSON DATA INSTANCE.
2. DO NOT ECHO OR COPY THE JSON SCHEMA.
3. DO NOT output schema keywords like "$defs", "type", "properties", or "required".
4. Provide REAL data that fills out the requested fields based on your research.

The JSON MUST strictly adhere to the shape defined by this JSON Schema:
{schema_json}

Rules:
- Do not wrap the JSON in markdown code blocks (e.g. ```json).
- Do not add explanations, conversational text, or <think> blocks outside the JSON.
- Every field in the schema MUST be present with actual values.
- If a value cannot be determined, use a schema-compatible fallback value (e.g. null or "").
- Ensure the JSON is strictly valid and parseable by standard JSON parsers.
""".strip()


def parse_structured_output(
    message: str,
    schema,
    llm: BaseChatModel | None = None,
):
    """
    Parse an LLM output into a Pydantic model.

    Steps:
        1. Direct JSON parse
        2. Bracket extraction
        3. Optional LLM repair (LLM must produce only JSON)
    """

    # ---- 1. Direct JSON load ----
    try:
        data = json.loads(message)
        try:
            return schema(**data)
        except Exception:
            data = _normalize_schema_echo(data, schema)
            return schema(**data)
    except Exception:
        pass

    # ---- 2. Naive bracket extraction ----
    try:
        start = message.index("{")
        end = message.rindex("}") + 1
        candidate = message[start:end]
        data = json.loads(candidate)
        try:
            return schema(**data)
        except Exception:
            data = _normalize_schema_echo(data, schema)
            return schema(**data)
    except Exception:
        pass

    # ---- 3. Optional LLM repair ----
    if llm:
        try:
            repair_prompt = f"""
            The following content is intended to be valid JSON matching this schema:

            {schema.model_json_schema()}

            Please fix the JSON structure. 
            Respond ONLY with valid JSON. Do not add explanations.

            Content:
            {message}
            """

            resp = llm.invoke(repair_prompt)
            fixed = resp.content if hasattr(resp, "content") else resp
            fixed = _coerce_message_text(fixed)
            data = json.loads(fixed)
            data = _normalize_schema_echo(data, schema)
            return schema(**data)

        except Exception:
            pass

    # ---- 4. Final error ----
    raise ValueError(
        f"Could not parse message into schema {schema.__name__}: {message}"
    )


def structured_output_repair_fn(llm: BaseChatModel, text: str) -> str:
    """
    Use an LLM to repair broken JSON in text.
    Args:
        llm: LLM instance
        text: raw text containing broken JSON
    Returns:
        Repaired JSON string"""
    prompt = f"Fix the following to valid JSON. Respond ONLY with JSON:\n{text}"
    
    response = llm.invoke([SystemMessage(content=prompt)])
    logger.debug(f"LLM repaired JSON: {response}")
    
    # Handle case where content might be a list
    if isinstance(response.content, list):
        return str(response.content[0]) if response.content else ""
    return response.content
