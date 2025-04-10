from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


@pytest.mark.usefixtures("test_global_and_local")
def test_categorical_lexical_sort() -> None:
    df = pl.DataFrame(
        {"cats": ["z", "z", "k", "a", "b"], "vals": [3, 1, 2, 2, 3]}
    ).with_columns(
        pl.col("cats").cast(pl.Categorical("lexical")),
    )

    out = df.sort(["cats"])
    assert out["cats"].dtype == pl.Categorical
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 3, 1]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.String)), expected)
    out = df.sort(["cats", "vals"])
    expected = pl.DataFrame(
        {"cats": ["a", "b", "k", "z", "z"], "vals": [2, 3, 2, 1, 3]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.String)), expected)
    out = df.sort(["vals", "cats"])

    expected = pl.DataFrame(
        {"cats": ["z", "a", "k", "b", "z"], "vals": [1, 2, 2, 3, 3]}
    )
    assert_frame_equal(out.with_columns(pl.col("cats").cast(pl.String)), expected)

    s = pl.Series(["a", "c", "a", "b", "a"], dtype=pl.Categorical("lexical"))
    assert s.sort().cast(pl.String).to_list() == [
        "a",
        "a",
        "a",
        "b",
        "c",
    ]


@pytest.mark.usefixtures("test_global_and_local")
def test_categorical_lexical_ordering_after_concat() -> None:
    ldf1 = (
        pl.DataFrame([pl.Series("key1", [8, 5]), pl.Series("key2", ["fox", "baz"])])
        .lazy()
        .with_columns(pl.col("key2").cast(pl.Categorical("lexical")))
    )
    ldf2 = (
        pl.DataFrame(
            [pl.Series("key1", [6, 8, 6]), pl.Series("key2", ["fox", "foo", "bar"])]
        )
        .lazy()
        .with_columns(pl.col("key2").cast(pl.Categorical("lexical")))
    )
    df = pl.concat([ldf1, ldf2]).select(pl.col("key2")).collect()

    assert df.sort("key2").to_dict(as_series=False) == {
        "key2": ["bar", "baz", "foo", "fox", "fox"]
    }


@pytest.mark.usefixtures("test_global_and_local")
@pytest.mark.may_fail_auto_streaming
def test_sort_categoricals_6014_internal() -> None:
    # create basic categorical
    df = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
        pl.col("key").cast(pl.Categorical)
    )

    out = df.sort("key")
    assert out.to_dict(as_series=False) == {"key": ["bbb", "aaa", "ccc"]}


@pytest.mark.usefixtures("test_global_and_local")
def test_sort_categoricals_6014_lexical() -> None:
    # create lexically-ordered categorical
    df = pl.DataFrame({"key": ["bbb", "aaa", "ccc"]}).with_columns(
        pl.col("key").cast(pl.Categorical("lexical"))
    )

    out = df.sort("key")
    assert out.to_dict(as_series=False) == {"key": ["aaa", "bbb", "ccc"]}


@pytest.mark.usefixtures("test_global_and_local")
def test_categorical_get_categories() -> None:
    assert pl.Series(
        "cats", ["foo", "bar", "foo", "foo", "ham"], dtype=pl.Categorical
    ).cat.get_categories().to_list() == ["foo", "bar", "ham"]


def test_cat_to_local() -> None:
    with pl.StringCache():
        s1 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
        s2 = pl.Series(["c", "b", "d"], dtype=pl.Categorical)

    # s2 physical starts after s1
    assert s1.to_physical().to_list() == [0, 1, 0]
    assert s2.to_physical().to_list() == [2, 1, 3]

    out = s2.cat.to_local()

    # Physical has changed and now starts at 0, string values are the same
    assert out.cat.is_local()
    assert out.to_physical().to_list() == [0, 1, 2]
    assert out.to_list() == s2.to_list()

    # s2 should be unchanged after the operation
    assert not s2.cat.is_local()
    assert s2.to_physical().to_list() == [2, 1, 3]
    assert s2.to_list() == ["c", "b", "d"]


def test_cat_to_local_missing_values() -> None:
    with pl.StringCache():
        _ = pl.Series(["a", "b"], dtype=pl.Categorical)
        s = pl.Series(["c", "b", None, "d"], dtype=pl.Categorical)

    out = s.cat.to_local()
    assert out.to_physical().to_list() == [0, 1, None, 2]


def test_cat_to_local_already_local() -> None:
    s = pl.Series(["a", "c", "a", "b"], dtype=pl.Categorical)

    assert s.cat.is_local()
    out = s.cat.to_local()

    assert out.to_physical().to_list() == [0, 1, 0, 2]
    assert out.to_list() == ["a", "c", "a", "b"]


def test_cat_is_local() -> None:
    s = pl.Series(["a", "c", "a", "b"], dtype=pl.Categorical)
    assert s.cat.is_local()

    with pl.StringCache():
        s2 = pl.Series(["a", "b", "a"], dtype=pl.Categorical)
    assert not s2.cat.is_local()


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_uses_lexical_ordering() -> None:
    s = pl.Series(["a", "b", None, "b"]).cast(pl.Categorical)
    assert s.cat.uses_lexical_ordering() is False

    s = s.cast(pl.Categorical("lexical"))
    assert s.cat.uses_lexical_ordering() is True

    s = s.cast(pl.Categorical("physical"))
    assert s.cat.uses_lexical_ordering() is False


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_len_bytes() -> None:
    # test Series
    s = pl.Series("a", ["Café", None, "Café", "345", "東京"], dtype=pl.Categorical)
    result = s.cat.len_bytes()
    expected = pl.Series("a", [5, None, 5, 3, 6], dtype=pl.UInt32)
    assert_series_equal(result, expected)

    # test DataFrame expr
    df = pl.DataFrame(s)
    result_df = df.select(pl.col("a").cat.len_bytes())
    expected_df = pl.DataFrame(expected)
    assert_frame_equal(result_df, expected_df)

    # test LazyFrame expr
    result_lf = df.lazy().select(pl.col("a").cat.len_bytes()).collect()
    assert_frame_equal(result_lf, expected_df)

    # test GroupBy
    result_df = (
        pl.LazyFrame({"key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], "value": s.extend(s)})
        .group_by("key", maintain_order=True)
        .agg(pl.col("value").cat.len_bytes().alias("len_bytes"))
        .explode("len_bytes")
        .collect()
    )
    expected_df = pl.DataFrame(
        {
            "key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "len_bytes": pl.Series(
                [5, None, 5, 3, 6, 5, None, 5, 3, 6], dtype=pl.get_index_type()
            ),
        }
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_len_chars() -> None:
    # test Series
    s = pl.Series("a", ["Café", None, "Café", "345", "東京"], dtype=pl.Categorical)
    result = s.cat.len_chars()
    expected = pl.Series("a", [4, None, 4, 3, 2], dtype=pl.UInt32)
    assert_series_equal(result, expected)

    # test DataFrame expr
    df = pl.DataFrame(s)
    result_df = df.select(pl.col("a").cat.len_chars())
    expected_df = pl.DataFrame(expected)
    assert_frame_equal(result_df, expected_df)

    # test LazyFrame expr
    result_lf = df.lazy().select(pl.col("a").cat.len_chars()).collect()
    assert_frame_equal(result_lf, expected_df)

    # test GroupBy
    result_df = (
        pl.LazyFrame({"key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], "value": s.extend(s)})
        .group_by("key", maintain_order=True)
        .agg(pl.col("value").cat.len_chars().alias("len_bytes"))
        .explode("len_bytes")
        .collect()
    )
    expected_df = pl.DataFrame(
        {
            "key": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "len_bytes": pl.Series(
                [4, None, 4, 3, 2, 4, None, 4, 3, 2], dtype=pl.get_index_type()
            ),
        }
    )
    assert_frame_equal(result_df, expected_df)


@pytest.mark.usefixtures("test_global_and_local")
def test_starts_ends_with() -> None:
    s = pl.Series(
        "a",
        ["hamburger_with_tomatoes", "nuts", "nuts", "lollypop", None],
        dtype=pl.Categorical,
    )
    assert_series_equal(
        s.cat.ends_with("pop"), pl.Series("a", [False, False, False, True, None])
    )
    assert_series_equal(
        s.cat.starts_with("nu"), pl.Series("a", [False, True, True, False, None])
    )

    with pytest.raises(TypeError, match="'prefix' must be a string; found"):
        s.cat.starts_with(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'suffix' must be a string; found"):
        s.cat.ends_with(None)  # type: ignore[arg-type]

    df = pl.DataFrame(
        {
            "a": pl.Series(
                ["hamburger_with_tomatoes", "nuts", "nuts", "lollypop", None],
                dtype=pl.Categorical,
            ),
        }
    )

    expected = {
        "ends_pop": [False, False, False, True, None],
        "starts_ham": [True, False, False, False, None],
    }

    assert (
        df.select(
            pl.col("a").cat.ends_with("pop").alias("ends_pop"),
            pl.col("a").cat.starts_with("ham").alias("starts_ham"),
        ).to_dict(as_series=False)
        == expected
    )

    with pytest.raises(TypeError, match="'prefix' must be a string; found"):
        df.select(pl.col("a").cat.starts_with(None))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="'suffix' must be a string; found"):
        df.select(pl.col("a").cat.ends_with(None))  # type: ignore[arg-type]


def test_cat_slice() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series(
                [
                    "foobar",
                    "barfoo",
                    "foobar",
                    "x",
                    None,
                ],
                dtype=pl.Categorical,
            )
        }
    )
    assert df["a"].cat.slice(-3).to_list() == ["bar", "foo", "bar", "x", None]
    assert df.select([pl.col("a").cat.slice(2, 4)])["a"].to_list() == [
        "obar",
        "rfoo",
        "obar",
        "",
        None,
    ]


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_to_lowercase() -> None:
    s = pl.Series(["Hello", "WORLD"], dtype=pl.Categorical)
    expected = pl.Series(["hello", "world"])
    assert_series_equal(s.cat.to_lowercase(), expected)


@pytest.mark.usefixtures("test_global_and_local")
def test_cat_to_uppercase() -> None:
    s = pl.Series(["Hello", "WORLD"], dtype=pl.Categorical)
    expected = pl.Series(["HELLO", "WORLD"])
    assert_series_equal(s.cat.to_uppercase(), expected)


def test_titlecase() -> None:
    df = pl.DataFrame(
        {
            "misc": pl.Series(
                [
                    "welcome to my world",
                    "double  space",
                    "and\ta\t tab",
                    "by jean-paul sartre, 'esq'",
                    "SOMETIMES/life/gives/you/a/2nd/chance",
                ],
                dtype=pl.Categorical,
            )
        }
    )
    expected = [
        "Welcome To My World",
        "Double  Space",
        "And\tA\t Tab",
        "By Jean-Paul Sartre, 'Esq'",
        "Sometimes/Life/Gives/You/A/2nd/Chance",
    ]
    actual = df.select(pl.col("misc").cat.to_titlecase()).to_series()
    for ex, act in zip(expected, actual):
        assert ex == act, f"{ex} != {act}"

    df = pl.DataFrame(
        {
            "quotes": pl.Series(
                [
                    "'e.t. phone home'",
                    "you talkin' to me?",
                    "i feel the need--the need for speed",
                    "to infinity,and BEYOND!",
                    "say 'what' again!i dare you - I\u00a0double-dare you!",
                    "What.we.got.here... is#failure#to#communicate",
                ],
                dtype=pl.Categorical,
            )
        }
    )
    expected_str = [
        "'E.T. Phone Home'",
        "You Talkin' To Me?",
        "I Feel The Need--The Need For Speed",
        "To Infinity,And Beyond!",
        "Say 'What' Again!I Dare You - I\u00a0Double-Dare You!",
        "What.We.Got.Here... Is#Failure#To#Communicate",
    ]
    expected_py = [s.title() for s in df["quotes"].to_list()]
    for ex_str, ex_py, act in zip(
        expected_str, expected_py, df["quotes"].cat.to_titlecase()
    ):
        assert ex_str == act, f"{ex_str} != {act}"
        assert ex_py == act, f"{ex_py} != {act}"


def test_sql_lowercase() -> None:
    dt = pl.Enum(["A", "B", "C", "D", "E"])
    df = pl.DataFrame(
        {
            "enum_col": pl.Series(["A", "B", "C", "D", "A"], dtype=dt),
            "int_col": [1, 2, 3, 4, 5],
        }
    )

    result = (
        pl.SQLContext({"sql_table": df})
        .execute("SELECT * FROM sql_table WHERE LOWER(enum_col) = 'a';")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "enum_col": pl.Series(["A", "A"], dtype=dt),
            "int_col": [1, 5],
        }
    )
    assert_frame_equal(result, expected)


def test_sql_uppercase() -> None:
    dt = pl.Enum(["a", "b", "c", "d", "e"])
    df = pl.DataFrame(
        {
            "enum_col": pl.Series(["a", "b", "c", "d", "a"], dtype=dt),
            "int_col": [1, 2, 3, 4, 5],
        }
    )

    result = (
        pl.SQLContext({"sql_table": df})
        .execute("SELECT * FROM sql_table WHERE UPPER(enum_col) = 'A';")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "enum_col": pl.Series(["a", "a"], dtype=dt),
            "int_col": [1, 5],
        }
    )
    assert_frame_equal(result, expected)


def test_sql_titlecase() -> None:
    dt = pl.Enum(["aa", "bb", "cc", "dd", "ee"])
    df = pl.DataFrame(
        {
            "enum_col": pl.Series(["aa", "bb", "cc", "dd", "aa"], dtype=dt),
            "int_col": [1, 2, 3, 4, 5],
        }
    )

    result = (
        pl.SQLContext({"sql_table": df})
        .execute("SELECT * FROM sql_table WHERE INITCAP(enum_col) = 'Aa';")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "enum_col": pl.Series(["aa", "aa"], dtype=dt),
            "int_col": [1, 5],
        }
    )
    assert_frame_equal(result, expected)
