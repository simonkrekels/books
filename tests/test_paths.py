from books.paths import render_template, slugify


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_punctuation():
    assert slugify("On the Theory of Quanta!") == "on-the-theory-of-quanta"


def test_slugify_unicode():
    assert slugify("Schrödinger") == "schrodinger"


def test_slugify_ampersand():
    assert slugify("Sakurai & Napolitano") == "sakurai-and-napolitano"


def test_slugify_truncates():
    long = "a" * 200
    assert len(slugify(long)) <= 80


def test_slugify_empty():
    assert slugify("") == "untitled"
    assert slugify("!!!") == "untitled"


def test_render_basic():
    paper = {
        "authors": [{"family": "Knuth", "given": "Donald E."}],
        "year": 1968,
        "title": "The Art of Computer Programming",
        "doi": "10.0001/foo",
        "journal": None,
    }
    out = render_template("{author_last}/{year}/{title_slug}.pdf", paper=paper)
    assert out == "knuth/1968/the-art-of-computer-programming.pdf"


def test_render_missing_year():
    paper = {"authors": [{"family": "Smith"}], "year": None, "title": "Foo"}
    out = render_template("{author_last}/{year}/{title_slug}.pdf", paper=paper)
    assert out == "smith/unknown/foo.pdf"


def test_render_no_authors():
    paper = {"authors": [], "year": 2020, "title": "Bar"}
    out = render_template("{author_last}/{year}/{title_slug}.pdf", paper=paper)
    assert out == "unknown/2020/bar.pdf"
