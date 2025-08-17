import argparse
import logging
from .logging_conf import setup_logging
from .data_loader import configure_pandas, load_transactions, load_fx
from .feature_engineering import convert_to_usd, build_features
from .eda import run_eda

def cmd_run_all():
    configure_pandas()
    tx = load_transactions()
    fx = load_fx()
    tx = convert_to_usd(tx, fx, strict=False)
    df = build_features(tx)
    out_dir = run_eda(df)
    logging.getLogger(__name__).info(f"Готово. Отчёты: {out_dir}")

def cmd_run_eda():
    configure_pandas()
    tx = load_transactions()
    out_dir = run_eda(tx)
    logging.getLogger(__name__).info(f"Готово. Отчёты: {out_dir}")

def cmd_run_fe():
    configure_pandas()
    tx = load_transactions()
    fx = load_fx()
    df = convert_to_usd(tx, fx, strict=False)
    df = build_features(df)
    out_dir = run_eda(df)
    logging.getLogger(__name__).info(f"Готово. Отчёты по фичам: {out_dir}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Fraud EDA/FE pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run-all", help="загрузка -> конвертация -> фичи -> EDA")
    sub.add_parser("run-eda", help="только EDA без конвертации/фич")
    sub.add_parser("run-fe", help="загрузка -> конвертация -> фичи -> EDA (по фичам)")

    args = parser.parse_args()
    if args.cmd == "run-all":
        cmd_run_all()
    elif args.cmd == "run-eda":
        cmd_run_eda()
    elif args.cmd == "run-fe":
        cmd_run_fe()

if __name__ == "__main__":
    main()
