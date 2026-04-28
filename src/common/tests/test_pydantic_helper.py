import json
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from src.common.pydantic_helper import (
    build_structured_prompt,
    parse_structured_output,
    _normalize_schema_echo,
    _recover_schema_value,
    _coerce_message_text,
)


class PersonSchema(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


class TagsSchema(BaseModel):
    tags: list[str] = Field(description="A list of tags")


class TestBuildStructuredPrompt:

    def test_includes_original_prompt(self):
        result = build_structured_prompt("Who is Alice?", PersonSchema)
        assert "Who is Alice?" in result

    def test_includes_json_schema(self):
        result = build_structured_prompt("test", PersonSchema)
        schema_json = json.dumps(PersonSchema.model_json_schema(), indent=2)
        assert schema_json in result

    def test_includes_json_only_instruction(self):
        result = build_structured_prompt("test", PersonSchema)
        assert "ONLY with a single valid JSON object" in result

    def test_no_code_block_instruction(self):
        result = build_structured_prompt("test", PersonSchema)
        assert "Do not wrap the JSON in code blocks" in result


class TestParseStructuredOutput:

    def test_direct_json_parse(self):
        message = '{"name": "Alice", "age": 30}'
        result = parse_structured_output(message, PersonSchema)
        assert result.name == "Alice"
        assert result.age == 30

    def test_bracket_extraction_with_preamble(self):
        message = 'Here is the result: {"name": "Bob", "age": 25} hope that helps!'
        result = parse_structured_output(message, PersonSchema)
        assert result.name == "Bob"
        assert result.age == 25

    def test_raises_when_unparseable_and_no_llm(self):
        with pytest.raises(ValueError, match="Could not parse message"):
            parse_structured_output("this is not json at all", PersonSchema)

    def test_llm_repair_fallback(self):
        broken = "name: Alice, age: 30"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content='{"name": "Alice", "age": 30}')

        result = parse_structured_output(broken, PersonSchema, llm=mock_llm)
        assert result.name == "Alice"
        assert result.age == 30
        mock_llm.invoke.assert_called_once()

    def test_llm_repair_also_fails_raises(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="still not json")

        with pytest.raises(ValueError, match="Could not parse message"):
            parse_structured_output("not json", PersonSchema, llm=mock_llm)

    def test_list_content_response(self):
        """Handles LLM responses where content is a list of dicts (e.g. Claude-style)."""
        message = '{"name": "Dan", "age": 22}'
        result = parse_structured_output(message, PersonSchema)
        assert result.name == "Dan"


class TestNormalizeSchemaEcho:

    def test_passthrough_clean_data(self):
        data = {"name": "Alice", "age": 30}
        result = _normalize_schema_echo(data, PersonSchema)
        assert result["name"] == "Alice"
        assert result["age"] == 30

    def test_unwraps_properties_key(self):
        """LLM echoes schema with 'properties' wrapper."""
        data = {
            "properties": {
                "name": "Eve",
                "age": 28
            }
        }
        result = _normalize_schema_echo(data, PersonSchema)
        assert result["name"] == "Eve"
        assert result["age"] == 28

    def test_non_dict_passthrough(self):
        assert _normalize_schema_echo("not a dict", PersonSchema) == "not a dict"


class TestRecoverSchemaValue:

    def test_plain_value_passthrough(self):
        assert _recover_schema_value("hello") == "hello"
        assert _recover_schema_value(42) == 42

    def test_extracts_value_key(self):
        assert _recover_schema_value({"value": "extracted"}) == "extracted"

    def test_extracts_enum_first(self):
        assert _recover_schema_value({"enum": ["first", "second"]}) == "first"

    def test_extracts_properties(self):
        data = {"properties": {"x": "a", "y": "b"}}
        result = _recover_schema_value(data)
        assert result == {"x": "a", "y": "b"}

    def test_list_recovery(self):
        result = _recover_schema_value([{"value": 1}, {"value": 2}])
        assert result == [1, 2]

    def test_returns_none_for_schema_only_dict(self):
        schema_dict = {"type": "string", "title": "Name", "description": "A name"}
        assert _recover_schema_value(schema_dict) is None


class TestCoerceMessageText:

    def test_string_passthrough(self):
        assert _coerce_message_text("hello") == "hello"

    def test_list_of_dicts_with_text(self):
        content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        assert _coerce_message_text(content) == "part1part2"

    def test_list_of_dicts_missing_text_key(self):
        content = [{"type": "image"}]
        assert _coerce_message_text(content) == ""

    def test_non_string_non_list_coerced(self):
        assert _coerce_message_text(123) == "123"
