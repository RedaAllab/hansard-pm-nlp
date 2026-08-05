from hansard_pm_nlp.cleaning import clean_contribution_text, clean_debate_section


def test_strips_em_tags():
    assert clean_contribution_text("This is <em>important</em>.") == "This is important."


def test_strips_column_number_span():
    text = 'Before <span id="25" class="column-number" data-column-number="25"> </span>after.'
    assert clean_contribution_text(text) == "Before after."


def test_strips_sub_and_strong_tags():
    assert clean_contribution_text("H<sub>2</sub>O and <strong>bold</strong>") == "H2O and bold"


def test_unescapes_html_entities():
    assert clean_contribution_text("Fish &amp; chips") == "Fish & chips"


def test_normalizes_crlf_and_collapses_whitespace():
    assert clean_contribution_text("Line one.\r\n\r\nLine two.") == "Line one. Line two."


def test_strips_leading_trailing_whitespace():
    assert clean_contribution_text("  padded text  ") == "padded text"


def test_clean_contribution_text_handles_plain_text_unchanged():
    assert clean_contribution_text("Nothing to clean here.") == "Nothing to clean here."


def test_clean_debate_section_strips_whitespace():
    assert clean_debate_section(" Covid-19 Update") == "Covid-19 Update"
    assert clean_debate_section("Covid-19 Update") == "Covid-19 Update"


def test_clean_debate_section_merges_whitespace_variants():
    variants = [" Covid-19 Update", "Covid-19 Update  ", "Covid-19 Update"]
    assert {clean_debate_section(v) for v in variants} == {"Covid-19 Update"}
