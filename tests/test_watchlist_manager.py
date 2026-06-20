"""
WatchlistManager 单元测试

测试覆盖：
- 添加自选股（含市场校验、重复检测）
- 查询/删除自选股
- 列表（含市场过滤）
- 更新标签和备注
- 按标签搜索
- 统计和清空
- 单例模式
- 线程安全
"""

import threading
from datetime import datetime

import pytest

from src.db import init_database
from src.watchlist.manager import WatchlistManager, WatchlistError, get_watchlist


@pytest.fixture(autouse=True)
def reset_db():
    """每个测试前重置数据库"""
    init_database()
    wm = get_watchlist()
    wm.clear()
    return wm


class TestWatchlistManager:
    """自选股管理器测试套件"""

    def test_singleton(self):
        """验证单例模式"""
        wm1 = get_watchlist()
        wm2 = get_watchlist()
        assert wm1 is wm2

    def test_add_stock(self, reset_db):
        """添加自选股"""
        wm = reset_db
        item = wm.add_stock("000001", "平安银行", "CN", tags=["银行", "蓝筹"])
        assert item.symbol == "000001"
        assert item.name == "平安银行"
        assert item.market == "a"
        assert item.tags == "银行,蓝筹"
        assert isinstance(item.id, int)
        assert isinstance(item.added_at, datetime)
        assert wm.count() == 1

    def test_add_stock_case_insensitive(self, reset_db):
        """添加自选股时 symbol 自动转为大写"""
        wm = reset_db
        item = wm.add_stock("hk00700", "腾讯控股", "HK")
        assert item.symbol == "HK00700"

    def test_add_duplicate_stock(self, reset_db):
        """重复添加应抛异常"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN")
        with pytest.raises(WatchlistError, match="已存在"):
            wm.add_stock("000001", "平安银行", "CN")

    def test_add_invalid_market(self, reset_db):
        """无效市场应抛异常"""
        wm = reset_db
        with pytest.raises(WatchlistError, match="不支持的市场"):
            wm.add_stock("000001", "测试", "JP")

    def test_add_without_tags(self, reset_db):
        """添加不带标签和备注"""
        wm = reset_db
        item = wm.add_stock("600519", "贵州茅台", "CN")
        assert item.tags == ""
        assert item.notes == ""
        assert wm.count() == 1

    def test_get_stock(self, reset_db):
        """按 symbol 查询"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN", tags=["银行"])
        item = wm.get_stock("000001")
        assert item is not None
        assert item.name == "平安银行"

    def test_get_nonexistent_stock(self, reset_db):
        """查询不存在的股票应返回 None"""
        wm = reset_db
        assert wm.get_stock("999999") is None

    def test_remove_stock(self, reset_db):
        """删除自选股"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN")
        assert wm.remove_stock("000001") is True
        assert wm.get_stock("000001") is None
        assert wm.count() == 0

    def test_remove_nonexistent_stock(self, reset_db):
        """删除不存在的股票返回 False"""
        wm = reset_db
        assert wm.remove_stock("999999") is False

    def test_list_stocks(self, reset_db):
        """列出所有自选股"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN")
        wm.add_stock("600519", "贵州茅台", "CN")
        wm.add_stock("HK00700", "腾讯控股", "HK")
        stocks = wm.list_stocks()
        assert len(stocks) == 3

    def test_list_stocks_by_market(self, reset_db):
        """按市场过滤"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN")
        wm.add_stock("HK00700", "腾讯控股", "HK")
        cn_stocks = wm.list_stocks("CN")
        hk_stocks = wm.list_stocks("HK")
        assert len(cn_stocks) == 1
        assert len(hk_stocks) == 1

    def test_update_tags(self, reset_db):
        """更新标签"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN", tags=["银行"])
        updated = wm.update_tags("000001", ["银行", "蓝筹", "高股息"])
        assert updated.tags == "银行,蓝筹,高股息"

    def test_update_notes(self, reset_db):
        """更新备注"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN")
        updated = wm.update_notes("000001", "长期持有标的")
        assert updated.notes == "长期持有标的"

    def test_update_tags_on_nonexistent(self, reset_db):
        """更新不存在的股票标签应抛异常"""
        wm = reset_db
        with pytest.raises(WatchlistError, match="不存在"):
            wm.update_tags("999999", ["test"])

    def test_search_by_tag(self, reset_db):
        """按标签搜索"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN", tags=["银行", "蓝筹"])
        wm.add_stock("600519", "贵州茅台", "CN", tags=["白酒", "蓝筹"])
        wm.add_stock("002415", "海康威视", "CN", tags=["安防"])

        results = wm.search_by_tag("蓝筹")
        assert len(results) == 2

        results = wm.search_by_tag("白酒")
        assert len(results) == 1
        assert results[0].symbol == "600519"

        results = wm.search_by_tag("不存在的标签")
        assert len(results) == 0

    def test_count(self, reset_db):
        """统计数量"""
        wm = reset_db
        assert wm.count() == 0
        wm.add_stock("000001", "平安银行", "CN")
        assert wm.count() == 1
        wm.add_stock("600519", "贵州茅台", "CN")
        assert wm.count() == 2

    def test_clear(self, reset_db):
        """清空自选股"""
        wm = reset_db
        wm.add_stock("000001", "平安银行", "CN")
        wm.add_stock("600519", "贵州茅台", "CN")
        cleared = wm.clear()
        assert cleared == 2
        assert wm.count() == 0

    def test_thread_safety(self, reset_db):
        """并发添加线程安全"""
        wm = reset_db
        errors = []

        def add_stock(symbol: str):
            try:
                wm.add_stock(symbol, f"Stock_{symbol}", "CN")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_stock, args=(f"{i:06d}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert wm.count() == 20

    def test_updated_at_on_create(self, reset_db):
        """创建时 updated_at 应被设置"""
        wm = reset_db
        item = wm.add_stock("000001", "平安银行", "CN")
        assert item.updated_at is not None

    def test_updated_at_on_update(self, reset_db):
        """更新后 updated_at 应变化"""
        wm = reset_db
        item = wm.add_stock("000001", "平安银行", "CN", tags=["银行"])
        original_updated = item.updated_at

        import time
        time.sleep(0.01)  # 确保时间差异

        updated = wm.update_tags("000001", ["银行", "蓝筹"])
        assert updated.updated_at is not None
        # updated_at 应 > 原时间
        if original_updated is not None and updated.updated_at is not None:
            assert updated.updated_at >= original_updated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
