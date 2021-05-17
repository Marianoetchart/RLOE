from copy import deepcopy


class Agent:

    def __init__(self, id, name, type, random_state, log_to_file=True):

        self.id = id
        self.name = name
        self.type = type
        self.log_to_file = log_to_file
        self.random_state = random_state

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required " + "for every agents.Agent", self.name)

        self.kernel = None
        self.currentTime = None
        self.log = []
        self.logEvent("AGENT_TYPE", type)

    def kernelInitializing(self, kernel):
        self.kernel = kernel

    def kernelStarting(self, startTime):
        self.setWakeup(startTime)

    def kernelStopping(self):
        pass

    def kernelTerminating(self):
        if self.log and self.log_to_file:
            self.writeLog(self.log)

    def logEvent(self, eventType, event='', appendSummaryLog=False):
        e = deepcopy(event)
        self.log.append({'EventTime': self.currentTime, 'EventType': eventType, 'Event': e})
        if appendSummaryLog: self.kernel.appendSummaryLog(self.id, eventType, e)

    def receiveMessage(self, currentTime, msg):
        self.currentTime = currentTime

    def wakeup(self, currentTime):
        self.currentTime = currentTime

    def sendMessage(self, recipientID, msg):
        self.kernel.sendMessage(self.id, recipientID, msg)

    def setWakeup(self, requestedTime):
        self.kernel.setWakeup(self.id, requestedTime)

    def writeLog(self, log, filename=None):
        self.kernel.writeLog(self.id, log, filename)

    def updateAgentState(self, state):
        self.kernel.updateAgentState(self.id, state)

    def __lt__(self, other):
        return "{}".format(self.id) < "{}".format(other.id)
