"""Tests for nested section support in Prompt Poet."""

import pytest
from prompt import Prompt, PromptPart, PromptSection


class TestSectionParsing:
    """Test basic section parsing functionality."""

    def test_section_parsing_basic(self):
        """Test that sections are parsed and content is concatenated correctly."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        assert len(prompt.parts) == 1
        part = prompt.parts[0]
        assert part.name == "test_part"
        assert part.content == "HelloWorld"
        assert part.sections is not None
        assert len(part.sections) == 2
        assert part.sections[0].name == "section1"
        assert part.sections[0].content == "Hello"
        assert part.sections[1].name == "section2"
        assert part.sections[1].content == "World"

    def test_section_with_whitespace(self):
        """Test that whitespace is handled correctly in sections."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: |
        Hello
        World
    - name: section2
      content: |
        Foo
        Bar
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        part = prompt.parts[0]
        # Individual sections preserve trailing newlines from block scalars,
        # except the last section which loses its trailing newline due to Jinja2's default behavior
        assert part.sections[0].content == "Hello\nWorld\n"
        assert part.sections[1].content == "Foo\nBar"
        # Concatenated content has newlines between sections
        assert part.content == "Hello\nWorld\nFoo\nBar"

    def test_section_with_newline_separation(self):
        """Test that sections are separated by newlines from block scalars."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: |
        Hello
    - name: section2
      content: |
        World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        part = prompt.parts[0]
        assert part.sections[0].content == "Hello\n"
        assert part.sections[1].content == "World"
        assert part.content == "Hello\nWorld"

    def test_section_with_jinja2_variables(self):
        """Test that Jinja2 variables work in sections with newline separation."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: |
        Hello {{ name }}
    - name: section2
      content: |
        You are {{ age }} years old
"""
        prompt = Prompt(
            raw_template=raw_template,
            template_data={"name": "Alice", "age": 30}
        )

        part = prompt.parts[0]
        assert part.sections[0].content == "Hello Alice\n"
        assert part.sections[1].content == "You are 30 years old"
        assert part.content == "Hello Alice\nYou are 30 years old"


class TestSectionTokenization:
    """Test section tokenization functionality."""

    def test_section_tokenization(self):
        """Test that sections are tokenized independently."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        part = prompt.parts[0]
        assert part.tokens is not None
        assert len(part.tokens) > 0

        assert part.sections[0].tokens is not None
        assert len(part.sections[0].tokens) > 0
        assert part.sections[1].tokens is not None
        assert len(part.sections[1].tokens) > 0


class TestSectionStatistics:
    """Test section statistics API."""

    def test_section_stats(self):
        """Test section_stats property returns correct structure."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        stats = prompt.section_stats
        assert len(stats) == 1
        assert stats[0]["part_name"] == "test_part"
        assert stats[0]["part_tokens"] > 0
        assert stats[0]["part_role"] == "user"
        assert stats[0]["has_sections"] is True
        assert len(stats[0]["sections"]) == 2
        assert stats[0]["sections"][0]["section_name"] == "section1"
        assert stats[0]["sections"][0]["section_tokens"] > 0
        assert stats[0]["sections"][1]["section_name"] == "section2"
        assert stats[0]["sections"][1]["section_tokens"] > 0

    def test_get_section_token_counts(self):
        """Test get_section_token_counts helper method."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        counts = prompt.get_section_token_counts()
        assert "test_part" in counts
        assert "section1" in counts["test_part"]
        assert "section2" in counts["test_part"]
        assert counts["test_part"]["section1"] > 0
        assert counts["test_part"]["section2"] > 0

    def test_section_stats_without_sections(self):
        """Test section_stats with parts that don't have sections."""
        raw_template = """
- name: test_part
  content: Hello World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        stats = prompt.section_stats
        assert len(stats) == 1
        assert stats[0]["part_name"] == "test_part"
        assert stats[0]["has_sections"] is False
        assert stats[0]["sections"] == []

    def test_get_section_token_counts_without_sections(self):
        """Test get_section_token_counts with parts without sections."""
        raw_template = """
- name: test_part
  content: Hello World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        counts = prompt.get_section_token_counts()
        assert "test_part" in counts
        assert "_part_total" in counts["test_part"]
        assert counts["test_part"]["_part_total"] > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing templates."""

    def test_backward_compatibility_content_only(self):
        """Test that parts with only content field work unchanged."""
        raw_template = """
- name: test_part
  content: Hello World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        part = prompt.parts[0]
        assert part.name == "test_part"
        assert part.content == "Hello World"
        assert part.sections is None
        assert part.tokens is not None

    def test_sections_produce_identical_output(self):
        """Test that templates with and without sections produce identical output.

        Note: Sections use newlines for separation via YAML block scalars.
        """
        # Template without sections
        template_without_sections = """
- name: system_instructions
  role: system
  content: |
    Your name is Alice and you are a helpful assistant.
    Never be harmful to humans. Always be respectful and kind.
    Keep your responses concise and engaging.
- name: user_query
  role: user
  content: Hello!
"""
        # Template with sections using newlines for separation
        template_with_sections = """
- name: system_instructions
  role: system
  sections:
    - name: character_intro
      content: |
        Your name is Alice and you are a helpful assistant.
    - name: safety_rules
      content: |
        Never be harmful to humans. Always be respectful and kind.
    - name: conversation_style
      content: |
        Keep your responses concise and engaging.
- name: user_query
  role: user
  content: Hello!
"""
        prompt_without = Prompt(raw_template=template_without_sections, template_data={})
        prompt_with = Prompt(raw_template=template_with_sections, template_data={})

        # Test that string output is identical
        assert prompt_without.string == prompt_with.string

        # Test that messages are identical
        assert prompt_without.messages == prompt_with.messages

        # Tokenize and test token output
        prompt_without.tokenize()
        prompt_with.tokenize()

        # Tokens should be identical
        assert prompt_without.tokens == prompt_with.tokens

        # Total token count should be identical
        assert len(prompt_without.tokens) == len(prompt_with.tokens)

        # Parts should have same content
        assert len(prompt_without.parts) == len(prompt_with.parts)
        for part_without, part_with in zip(prompt_without.parts, prompt_with.parts):
            assert part_without.content == part_with.content
            assert part_without.role == part_with.role

    def test_mixed_parts_sections(self):
        """Test mixing parts with and without sections."""
        raw_template = """
- name: part1
  content: Hello
- name: part2
  sections:
    - name: section1
      content: World
    - name: section2
      content: Test
- name: part3
  content: End
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        assert len(prompt.parts) == 3
        assert prompt.parts[0].sections is None
        assert prompt.parts[1].sections is not None
        assert len(prompt.parts[1].sections) == 2
        assert prompt.parts[2].sections is None


class TestValidation:
    """Test validation of section configurations."""

    def test_empty_sections_error(self):
        """Test that empty sections list raises error."""
        raw_template = """
- name: test_part
  sections: []
"""
        with pytest.raises(ValueError, match="empty sections field"):
            Prompt(raw_template=raw_template, template_data={})

    def test_both_content_and_sections_error(self):
        """Test that having both content and sections raises error."""
        raw_template = """
- name: test_part
  content: Hello
  sections:
    - name: section1
      content: World
"""
        with pytest.raises(ValueError, match="cannot have both 'content' and 'sections'"):
            Prompt(raw_template=raw_template, template_data={})

    def test_section_missing_name(self):
        """Test that missing section name raises error."""
        raw_template = """
- name: test_part
  sections:
    - content: Hello
"""
        with pytest.raises(ValueError, match="missing 'name' field"):
            Prompt(raw_template=raw_template, template_data={})

    def test_section_missing_content(self):
        """Test that missing section content raises error."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
"""
        with pytest.raises(ValueError, match="missing 'content' field"):
            Prompt(raw_template=raw_template, template_data={})

    def test_sections_not_list(self):
        """Test that sections must be a list."""
        raw_template = """
- name: test_part
  sections: "not a list"
"""
        with pytest.raises(ValueError, match="sections must be a list"):
            Prompt(raw_template=raw_template, template_data={})

    def test_section_not_dict(self):
        """Test that each section must be a dict."""
        raw_template = """
- name: test_part
  sections:
    - "not a dict"
"""
        with pytest.raises(ValueError, match="Section .* must be a dict"):
            Prompt(raw_template=raw_template, template_data={})


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_section_with_special_characters(self):
        """Test sections with special characters."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: |
        Line 1
        Line 2
    - name: section2
      content: "Tab\\tNewline\\n"
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        part = prompt.parts[0]
        assert "Line 1" in part.sections[0].content
        assert "Line 2" in part.sections[0].content

    def test_section_with_role(self):
        """Test that role is preserved with sections."""
        raw_template = """
- name: test_part
  role: assistant
  sections:
    - name: section1
      content: Hello
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        part = prompt.parts[0]
        assert part.role == "assistant"
        assert part.sections is not None

    def test_section_with_truncation_priority(self):
        """Test that truncation_priority is preserved with sections."""
        raw_template = """
- name: test_part
  truncation_priority: 5
  sections:
    - name: section1
      content: Hello
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        part = prompt.parts[0]
        assert part.truncation_priority == 5
        assert part.sections is not None

    def test_section_level_truncation_priority(self):
        """Test section-level truncation priority (for future use)."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
      truncation_priority: 10
    - name: section2
      content: World
      truncation_priority: 5
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        part = prompt.parts[0]
        assert part.sections[0].truncation_priority == 10
        assert part.sections[1].truncation_priority == 5

    def test_empty_section_content(self):
        """Test section with empty content."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: ""
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        part = prompt.parts[0]
        assert part.sections[0].content == ""
        assert part.sections[1].content == "World"


class TestIntegration:
    """Integration tests with truncation and other features."""

    def test_sections_preserved_in_backup(self):
        """Test that sections are preserved in parts backup."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        # Access pretruncation parts (creates backup)
        assert prompt.pretruncation_parts[0].sections is not None
        assert len(prompt.pretruncation_parts[0].sections) == 2

    def test_sections_with_multiple_parts(self):
        """Test sections work correctly with multiple parts."""
        raw_template = """
- name: part1
  sections:
    - name: sec1
      content: A
    - name: sec2
      content: B
- name: part2
  sections:
    - name: sec3
      content: C
    - name: sec4
      content: D
"""
        prompt = Prompt(raw_template=raw_template, template_data={})
        prompt.tokenize()

        stats = prompt.section_stats
        assert len(stats) == 2
        assert len(stats[0]["sections"]) == 2
        assert len(stats[1]["sections"]) == 2

    def test_sections_with_messages_property(self):
        """Test that messages property works with sections."""
        raw_template = """
- name: test_part
  role: system
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        messages = prompt.messages
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "HelloWorld"

    def test_sections_with_string_property(self):
        """Test that string property works with sections."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: Hello
    - name: section2
      content: World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        assert prompt.string == "HelloWorld"

    def test_sections_with_new_lines(self):
        """Test that sections can have extra newlines for spacing."""
        raw_template = """
- name: test_part
  sections:
    - name: section1
      content: |+
        Hello

    - name: section2
      content: >-

        World
"""
        prompt = Prompt(raw_template=raw_template, template_data={})

        # Verify sections preserve newlines
        assert prompt.parts[0].sections[0].content == "Hello\n\n"
        assert prompt.parts[0].sections[1].content == "\nWorld"
        # Concatenated content maintains newline structure
        assert prompt.string == "Hello\n\n\nWorld"
