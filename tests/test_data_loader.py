import numpy as np
import pytest

from trading_bot.data_loader import (
    Dataset,
    generate_synthetic_data,
    load_kaggle_btc_data,
    load_or_generate_data,
)


class TestGenerateSyntheticData:
    def test_returns_dataset(self):
        ds = generate_synthetic_data(n_train=100, n_val=50, n_test=50, seed=42)
        assert isinstance(ds, Dataset)

    def test_correct_lengths(self):
        ds = generate_synthetic_data(n_train=200, n_val=50, n_test=100, seed=42)
        assert len(ds.train_prices) == 200
        assert len(ds.val_prices) == 50
        assert len(ds.test_prices) == 100

    def test_prices_positive(self):
        ds = generate_synthetic_data(seed=42)
        assert np.all(ds.train_prices > 0)
        assert np.all(ds.val_prices > 0)
        assert np.all(ds.test_prices > 0)

    def test_reproducibility(self):
        ds1 = generate_synthetic_data(seed=42)
        ds2 = generate_synthetic_data(seed=42)
        np.testing.assert_array_equal(ds1.train_prices, ds2.train_prices)
        np.testing.assert_array_equal(ds1.val_prices, ds2.val_prices)
        np.testing.assert_array_equal(ds1.test_prices, ds2.test_prices)

    def test_train_years_property(self):
        ds = generate_synthetic_data()
        assert ds.train_years == (2017, 2017)

    def test_test_years_property(self):
        ds = generate_synthetic_data()
        assert ds.test_years == (2021, 2021)

    def test_val_years_property(self):
        ds = generate_synthetic_data()
        assert ds.val_years == (2019, 2019)


class TestLoadKaggleBtcData:
    def test_missing_file_raises(self, tmp_path):
        missing_csv = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_kaggle_btc_data(missing_csv)

    def test_loads_valid_csv(self, tmp_path):
        csv = tmp_path / "btc.csv"
        csv.write_text(
            "Date,Open,High,Low,Close,Volume\n"
            "2016-01-01,100,110,90,105,1000\n"
            "2017-01-01,105,115,95,110,1200\n"
            "2018-01-01,110,120,100,115,1500\n"
            "2019-01-01,112,122,102,117,1600\n"
            "2020-01-01,200,220,180,210,2000\n"
            "2021-01-01,210,230,190,220,2500\n"
        )
        ds = load_kaggle_btc_data(
            csv,
            train_end="2017-12-31",
            val_start="2018-01-01",
            val_end="2019-12-31",
            test_start="2020-01-01",
        )

        assert len(ds.train_prices) == 2
        assert len(ds.val_prices) == 2
        assert len(ds.test_prices) == 2
        np.testing.assert_array_equal(ds.train_prices, [105.0, 110.0])
        np.testing.assert_array_equal(ds.val_prices, [115.0, 117.0])
        np.testing.assert_array_equal(ds.test_prices, [210.0, 220.0])

    def test_train_test_split_dates(self, tmp_path):
        csv = tmp_path / "btc.csv"
        csv.write_text(
            "Date,Close\n"
            "2016-12-30,100\n"
            "2017-12-31,101\n"
            "2018-01-01,102\n"
            "2019-12-31,103\n"
            "2020-01-01,104\n"
            "2021-01-01,105\n"
        )
        ds = load_kaggle_btc_data(
            csv,
            train_end="2017-12-31",
            val_start="2018-01-01",
            val_end="2019-12-31",
            test_start="2020-01-01",
        )

        assert len(ds.train_prices) == 2
        assert len(ds.val_prices) == 2
        assert len(ds.test_prices) == 2


class TestLoadOrGenerateData:
    def test_falls_back_to_synthetic(self, tmp_path):
        missing_csv = tmp_path / "missing.csv"
        ds = load_or_generate_data(missing_csv, fallback_to_synthetic=True)
        assert isinstance(ds, Dataset)
        assert len(ds.train_prices) > 0

    def test_raises_when_no_fallback(self, tmp_path):
        missing_csv = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            load_or_generate_data(missing_csv, fallback_to_synthetic=False)
