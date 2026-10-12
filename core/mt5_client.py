from __future__ import annotations

import MetaTrader5 as mt5


class MT5Client:
    """
    MetaTrader 5 bilan ulanishni boshqaradi.

    Vazifalari:
        - MT5 terminalga ulanish
        - terminal holatini tekshirish
        - symbol mavjudligini tekshirish
        - symbol ma'lumotlarini olish
        - MT5 ulanishini yopish
    """

    def connect(self) -> bool:
        """
        MT5 terminalga ulanadi.
        """

        # Agar MT5 allaqachon ishga tushgan bo'lsa,
        # qayta initialize qilishga harakat qilamiz.
        if not mt5.initialize():

            error = mt5.last_error()

            print(
                f"[MT5] Ulanish muvaffaqiyatsiz: {error}"
            )

            return False

        terminal = mt5.terminal_info()

        if terminal is None:

            print(
                "[MT5] Terminal ma'lumotlari olinmadi."
            )

            mt5.shutdown()

            return False

        print(
            "[MT5] Connected successfully."
        )

        print(
            f"[MT5] Terminal: {terminal.name}"
        )

        return True

    def shutdown(self) -> None:
        """
        MT5 ulanishini yopadi.
        """

        mt5.shutdown()

        print(
            "[MT5] Connection closed."
        )

    def symbol_exists(self, symbol: str) -> bool:
        """
        Symbol mavjudligini tekshiradi.
        """

        if not symbol:
            return False

        info = mt5.symbol_info(symbol)

        return info is not None

    def get_symbol_info(self, symbol: str):
        """
        Symbol haqida MT5 ma'lumotlarini qaytaradi.
        """

        if not symbol:
            return None

        return mt5.symbol_info(symbol)