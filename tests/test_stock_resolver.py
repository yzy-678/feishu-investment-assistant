"""StockResolver 单元测试。"""

from src.market.stock_resolver import StockResolver


def test_resolve_known_stock_names():
    resolver = StockResolver()

    assert resolver.resolve("有研新材").stock.symbol == "600206"
    assert resolver.resolve("中瓷电子").stock.symbol == "003031"
    assert resolver.resolve("长川科技").stock.symbol == "300604"
    assert resolver.resolve("富乐德").stock.symbol == "301297"
    assert resolver.resolve("超捷股份").stock.symbol == "301005"


def test_resolve_known_stock_code():
    stock = StockResolver().resolve("600206").stock

    assert stock.symbol == "600206"
    assert stock.name == "有研新材"
    assert stock.exchange == "SH"


def test_resolve_common_short_inputs():
    resolver = StockResolver()

    assert resolver.resolve("查一下有研新材").stock.symbol == "600206"
    assert resolver.resolve("分析中瓷电子").stock.symbol == "003031"
    assert resolver.resolve("看看长川科技").stock.symbol == "300604"
    assert resolver.resolve("有研新材怎么样").stock.symbol == "600206"


def test_resolve_unknown_stock_does_not_guess():
    result = StockResolver().resolve("不存在公司")

    assert result.stock is None
    assert result.candidates == ()
    assert result.error == "not_found"
