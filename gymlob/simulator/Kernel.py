<<<<<<< HEAD
import os, queue
from gymlob.simulator.util.Message import MessageType
from gymlob.simulator.util.util import pickle_data


class Kernel:

    def __init__(self, name, log_folder, skip_log=False, random_state=None):
        self.name = name
        self.log_folder = log_folder
        self.skip_log = skip_log
        self.random_state = random_state

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required for the Kernel")

        self.messages = queue.PriorityQueue()

        self.currentTime = None
        self.startTime = None
        self.stopTime = None

        self.agents = []
        self.agentCurrentTimes = []

        self.agentCountByType = {}
        self.summaryLog = []
        self.custom_state = {}

    def runner(self, agents=[], startTime=None, stopTime=None):

        self.agents = agents
        self.agentCurrentTimes = [startTime] * len(agents)

        self.startTime = startTime
        self.stopTime = stopTime

        for agent in self.agents:
            agent.kernelInitializing(self)

        for agent in self.agents:
            agent.kernelStarting(self.startTime)

        self.currentTime = self.startTime

        ttl_messages = 0

        while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):

            self.currentTime, event = self.messages.get()

            msg_recipient, msg_type, msg = event

            if ttl_messages % 100000 == 0:
                print("\n--- Simulation time: {}, messages processed: {} ---\n".format(self.currentTime, ttl_messages))

            agent = msg_recipient

            if self.agentCurrentTimes[agent] > self.currentTime:
                self.messages.put((self.agentCurrentTimes[agent], (msg_recipient, msg_type, msg)))
                continue

            self.agentCurrentTimes[agent] = self.currentTime

            ttl_messages += 1

            if msg_type == MessageType.WAKEUP:
                agents[agent].wakeup(self.currentTime)
            elif msg_type == MessageType.MESSAGE:
                agents[agent].receiveMessage(self.currentTime, msg)

        for agent in agents:
            agent.kernelStopping()

        for agent in agents:
            agent.kernelTerminating()

        self.writeSummaryLog()

        return self.custom_state

    def sendMessage(self, sender=None, recipient=None, msg=None):
        self.messages.put((self.currentTime, (recipient, MessageType.MESSAGE, msg)))

    def setWakeup(self, sender=None, requestedTime=None):
        self.messages.put((requestedTime, (sender, MessageType.WAKEUP, None)))

    def findAgentByType(self, type=None):

        for agent in self.agents:
            if isinstance(agent, type):
                return agent.id

    def writeLog(self, sender, log, filename=None):

        if self.skip_log: return

        if filename:
            file = "/{}.bz2".format(filename)
        else:
            file = "/{}.bz2".format(self.agents[sender].name.replace(" ", ""))

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        pickle_data(self.log_folder + file, log)

    def appendSummaryLog(self, sender, eventType, event):
        self.summaryLog.append({'AgentID': sender, 'EventType': eventType, 'Event': event})

    def writeSummaryLog(self):

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        filename = self.log_folder + "/summary_log.bz2"
        pickle_data(filename, self.summaryLog)

    def updateAgentState(self, agent_id, state):
        if 'agent_state' not in self.custom_state: self.custom_state['agent_state'] = {}
        self.custom_state['agent_state'][agent_id] = state
=======
import os, queue
from gymlob.simulator.util.Message import MessageType
from gymlob.simulator.util.util import pickle_data


class Kernel:

    def __init__(self, name, log_folder, skip_log=False, random_state=None):
        self.name = name
        self.log_folder = log_folder
        self.skip_log = skip_log
        self.random_state = random_state

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required for the Kernel")

        self.messages = queue.PriorityQueue()

        self.currentTime = None
        self.startTime = None
        self.stopTime = None

        self.agents = []
        self.agentCurrentTimes = []

        self.agentCountByType = {}
        self.summaryLog = []
        self.custom_state = {}

    def runner(self, agents=[], startTime=None, stopTime=None):

        self.agents = agents
        self.agentCurrentTimes = [startTime] * len(agents)

        self.startTime = startTime
        self.stopTime = stopTime

        for agent in self.agents:
            agent.kernelInitializing(self)

        for agent in self.agents:
            agent.kernelStarting(self.startTime)

        self.currentTime = self.startTime

        while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):

            self.currentTime, event = self.messages.get()

            msg_recipient, msg_type, msg = event

            agent = msg_recipient

            if self.agentCurrentTimes[agent] > self.currentTime:
                self.messages.put((self.agentCurrentTimes[agent], (msg_recipient, msg_type, msg)))
                continue

            self.agentCurrentTimes[agent] = self.currentTime

            if msg_type == MessageType.WAKEUP:
                agents[agent].wakeup(self.currentTime)
            elif msg_type == MessageType.MESSAGE:
                agents[agent].receiveMessage(self.currentTime, msg)

        for agent in agents:
            agent.kernelStopping()

        for agent in agents:
            agent.kernelTerminating()

        self.writeSummaryLog()

        return self.custom_state

    def sendMessage(self, sender=None, recipient=None, msg=None):
        self.messages.put((self.currentTime, (recipient, MessageType.MESSAGE, msg)))

    def setWakeup(self, sender=None, requestedTime=None):
        self.messages.put((requestedTime, (sender, MessageType.WAKEUP, None)))

    def findAgentByType(self, type=None):

        for agent in self.agents:
            if isinstance(agent, type):
                return agent.id

    def writeLog(self, sender, log, filename=None):

        if self.skip_log: return

        if filename:
            file = "/{}.bz2".format(filename)
        else:
            file = "/{}.bz2".format(self.agents[sender].name.replace(" ", ""))

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        pickle_data(self.log_folder + file, log)

    def appendSummaryLog(self, sender, eventType, event):
        self.summaryLog.append({'AgentID': sender, 'EventType': eventType, 'Event': event})

    def writeSummaryLog(self):

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        filename = self.log_folder + "/summary_log.bz2"
        pickle_data(filename, self.summaryLog)

    def updateAgentState(self, agent_id, state):
        if 'agent_state' not in self.custom_state: self.custom_state['agent_state'] = {}
        self.custom_state['agent_state'][agent_id] = state
>>>>>>> 2052760fb0c43ebd2c5008144699d9e0e9d2e88d
