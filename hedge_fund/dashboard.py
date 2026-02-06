"""
Trading dashboard for real-time position and candidate display.

Provides both Rich (colored, table-based) and ASCII fallback rendering
for the live bot's trading state.
"""


class Dashboard:
    """Real-time trading dashboard with Rich console or ASCII fallback."""

    def __init__(self):
        self.rich_available = False
        self._console = None
        self._box = None
        try:
            from rich.console import Console
            from rich import box
            self._console = Console()
            self._box = box
            self.rich_available = True
        except ImportError:
            pass

    def render(self, state):
        """
        Render the full dashboard state.

        Args:
            state: Dict with keys:
                equity (float): Current account equity.
                vix (float): Current VIX level.
                regime (str): Market regime (BULL/BEAR/NEUTRAL).
                universe_size (int): Number of tickers in universe.
                positions (list[dict]): Active positions, each with
                    symbol, side, qty, entry, curr, pnl_r, sl.
                candidates (list[dict]): Top trade opportunities, each with
                    symbol, score, p_win, ev, tier_mult, type.
                hedged (bool): Whether hedge position is active.
                pnl_day (float): Day's PnL.
                logs (list[str]): Recent event log lines.
        """
        if self.rich_available:
            self._render_rich(state)
        else:
            self._render_ascii(state)

    def _render_rich(self, state):
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text

        equity_color = "green" if state.get('pnl_day', 0) >= 0 else "red"
        regime_map = {"BULL": "B", "BEAR": "b", "NEUTRAL": "N"}
        regime_icon = {"BULL": "+", "BEAR": "-", "NEUTRAL": "~"}.get(
            state.get('regime', 'NEUTRAL'), '~'
        )

        metrics = [
            f"[bold cyan]Equity:[/bold cyan] [bold {equity_color}]${state['equity']:,.2f}[/]",
            f"[bold yellow]VIX:[/bold yellow] {state['vix']:.2f}",
            f"[bold magenta]Regime:[/bold magenta] {regime_icon} {state['regime']}",
            f"[bold blue]Univ:[/bold blue] {state['universe_size']}",
            f"[bold white]Hedged:[/bold white] {'Y' if state.get('hedged') else 'N'}",
        ]

        self._console.clear()
        self._console.rule("[bold gold1]GOD MODE v14.3[/bold gold1]")
        self._console.print(Panel(
            Columns([Text.from_markup(m) for m in metrics]),
            box=self._box.ROUNDED, expand=True,
        ))

        # Positions table
        pos_table = Table(
            title="[bold green]ACTIVE POSITIONS[/bold green]",
            box=self._box.SIMPLE_HEAD, expand=True,
        )
        pos_table.add_column("Sym", style="cyan")
        pos_table.add_column("Side", style="white")
        pos_table.add_column("Size", justify="right")
        pos_table.add_column("Entry", justify="right")
        pos_table.add_column("Curr", justify="right")
        pos_table.add_column("PnL (R)", justify="right")
        pos_table.add_column("Stop", justify="right")

        for p in state.get('positions', []):
            pnl_color = "green" if p.get('pnl_r', 0) > 0 else "red"
            pos_table.add_row(
                p['symbol'], p['side'], str(p['qty']),
                f"{p['entry']:.2f}", f"{p['curr']:.2f}",
                f"[{pnl_color}]{p['pnl_r']:.2f}R[/]",
                f"{p['sl']:.2f}",
            )

        if not state.get('positions'):
            pos_table.add_row("-", "-", "-", "-", "-", "-", "-")

        # Candidates table
        cand_table = Table(
            title="[bold yellow]TOP OPPORTUNITIES[/bold yellow]",
            box=self._box.SIMPLE_HEAD, expand=True,
        )
        cand_table.add_column("Sym", style="cyan")
        cand_table.add_column("Score", justify="right")
        cand_table.add_column("Win%", justify="right")
        cand_table.add_column("EV", justify="right")
        cand_table.add_column("Tier", style="magenta")
        cand_table.add_column("Type", style="white")

        for c in state.get('candidates', [])[:5]:
            cand_table.add_row(
                c['symbol'],
                f"{c['score']:.2f}",
                f"{c['p_win']:.0%}",
                f"{c['ev']:.2f}",
                f"{c['tier_mult']:.1f}x",
                c['type'],
            )

        if not state.get('candidates'):
            cand_table.add_row("-", "-", "-", "-", "-", "-")

        self._console.print(Columns([pos_table, cand_table], expand=True))

        if state.get('logs'):
            self._console.rule("[bold dim]Events[/bold dim]")
            for log in state['logs'][-5:]:
                self._console.print(f"[dim]{log}[/dim]")

    def _render_ascii(self, state):
        print("\n" + "=" * 60)
        print(
            f"GOD MODE v14.3 | ${state['equity']:,.2f} | "
            f"{state['regime']} | VIX: {state['vix']:.2f}"
        )
        print("-" * 60)
        print("POSITIONS:")
        print(f"{'SYM':<6} {'SIDE':<5} {'QTY':<5} {'ENTRY':<8} {'CURR':<8} {'PNL(R)':<6}")
        for p in state.get('positions', []):
            print(
                f"{p['symbol']:<6} {p['side']:<5} {p['qty']:<5} "
                f"{p['entry']:<8.2f} {p['curr']:<8.2f} {p['pnl_r']:<6.2f}"
            )
        print("-" * 60)
        print("TOP CANDIDATES:")
        print(f"{'SYM':<6} {'SCR':<5} {'WIN%':<5} {'EV':<5} {'TYPE'}")
        for c in state.get('candidates', [])[:5]:
            print(
                f"{c['symbol']:<6} {c['score']:<5.2f} {c['p_win']:<5.2f} "
                f"{c['ev']:<5.2f} {c['type']}"
            )
        print("=" * 60)

    def render_loading(self, message):
        """Show a loading/progress message."""
        print(f"  {message}")
