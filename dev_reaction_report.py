from worker_files import reactor_worker


"""
new_run = reactor_worker.VelmaReactor()
new_run.sort_files()
new_run.current_stats()
new_run.calc_rxn_rates()
new_run.plot_gas_rates()
"""
new_run = reactor_worker.VelmaReactor()
new_report = reactor_worker.ReactionReport(new_run)