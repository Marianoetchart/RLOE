import numpy as np
from datetime import datetime, timedelta
import logging as log

from gymlob.simulator.Kernel import Kernel


def abides(name: str,
           agents: list,
           date: str,
           stop_time: datetime,
           log_folder: str = None,
           skip_log: bool = False):

    simulation_start_time = datetime.now()

    kernel = Kernel(name, log_folder=log_folder+name, skip_log=skip_log,
                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    start_time = datetime.strptime(date, '%Y-%m-%d')
    stop_time = stop_time + timedelta(minutes=5)

    custom_state = kernel.runner(agents=agents, startTime=start_time, stopTime=stop_time)

    simulation_end_time = datetime.now()

    log.info("Time taken to run simulation {}: {}\n".format(name, simulation_end_time - simulation_start_time))

    return custom_state