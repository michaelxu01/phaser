import datetime
import logging
import typing as t

import pane
from pane.annotations import Tagged
from typing_extensions import Self

from .util import ReconsStateConverter


JobID: t.TypeAlias = str
WorkerID: t.TypeAlias = str

WorkerStatus: t.TypeAlias = t.Literal['queued', 'starting', 'reloading', 'idle', 'running', 'stopping', 'stopped', 'unknown']
"""
Enum indicating the state of the connection, as described by either the server or client.

- 'queued': Worker is queued/scheduled to start
- 'starting': Worker is actively being started
- 'reloading': Worker is being reloaded
- 'idle': No job is running
- 'running': Job is running
- 'stopping': Worker is being stopped
- 'stopped': Worker has been stopped (normally or abnormally)

- 'unknown': Job hasn't talked in a while, unknown state
"""

JobStatus: t.TypeAlias = t.Literal['queued', 'starting', 'running', 'stopping', 'stopped']
"""
Enum indicating the state of a reconstruction job
"""

Result: t.TypeAlias = t.Literal['finished', 'errored', 'cancelled', 'interrupted']
"""
Enum indicating a job or worker result.

- 'finished': Finished normally
- 'errored': Encounted an error running
- 'cancelled': Cancelled/shutdown by the user (from server side)
- 'interrupted': Cancelled/shutdown by the client/OS
"""

Signal: t.TypeAlias = t.Literal['shutdown', 'cancel', 'reload']
"""
Signal sent to a job or worker.
"""

class ValidationError(Exception):
    def __init__(self, msg: str):
        self.msg: str = msg

class SignalException(Exception):
    def __init__(self, signal: Signal, urgent: bool = False):
        self.signal: Signal = signal
        self.urgent: bool = urgent

# server -> client messages

class WorkerState(pane.PaneBase):
    worker_id: WorkerID
    worker_type: str
    status: WorkerStatus
    links: t.Dict[str, str] = pane.field(default_factory=dict)
    current_job: t.Optional[JobID] = None
    start_time: t.Optional[datetime.datetime] = None
    hostname: t.Optional[str] = None
    backends: t.Optional[t.Dict[str, t.Tuple[str, ...]]] = None

class WorkerUpdate(pane.PaneBase):
    worker_id: WorkerID
    status: WorkerStatus

    msg: t.Literal['status_change'] = 'status_change'

class JobState(pane.PaneBase):
    job_id: JobID
    status: JobStatus
    links: t.Dict[str, str] = pane.field(default_factory=dict)
    job_name: t.Optional[str] = None
    start_time: t.Optional[datetime.datetime] = None
    state: t.Dict[str, t.Any] = pane.field(converter=ReconsStateConverter(), default_factory=dict)

class LogRecord(pane.PaneBase):
    i: int

    timestamp: datetime.datetime

    log: str

    logger_name: str
    log_level: int

    line_number: int
    func_name: t.Optional[str] = None
    stack_info: t.Optional[str] = None

class JobStatusChange(pane.PaneBase):
    status: JobStatus
    job_id: JobID

    msg: t.Literal['status_change'] = 'status_change'

class JobUpdate(pane.PaneBase):
    state: t.Dict[str, t.Any] = pane.field(converter=ReconsStateConverter())
    job_id: JobID

    msg: t.Literal['job_update'] = 'job_update'

class LogUpdate(pane.PaneBase):
    new_logs: t.List[LogRecord]

    msg: t.Literal['log'] = 'log'

class JobStopped(pane.PaneBase):
    result: Result
    error: t.Optional[str] = None

    msg: t.Literal['job_stopped'] = 'job_stopped'

JobMessage: t.TypeAlias = t.Annotated[t.Union[
    JobStatusChange, JobUpdate, LogUpdate, JobStopped
], Tagged('msg')]

class DashboardConnected(pane.PaneBase):
    state: JobState
    msg: t.Literal['connected'] = 'connected'

DashboardMessage: t.TypeAlias = t.Annotated[t.Union[
    JobStatusChange, JobUpdate, JobStopped, DashboardConnected
], Tagged('msg')]

class WorkersUpdate(pane.PaneBase):
    event: t.Optional[WorkerUpdate]
    state: t.List[WorkerState]

    msg: t.Literal['workers_update'] = 'workers_update'

class JobsUpdate(pane.PaneBase):
    event: t.Optional[JobMessage]
    state: t.List[JobState]

    msg: t.Literal['jobs_update'] = 'jobs_update'

class ManagerConnected(pane.PaneBase):
    workers: t.List[WorkerState]
    jobs: t.List[JobState]

    msg: t.Literal['connected'] = 'connected'

ManagerMessage: t.TypeAlias = t.Annotated[t.Union[
    JobsUpdate, WorkersUpdate, ManagerConnected
], Tagged('msg')]

# worker -> server messages

class ConnectMessage(pane.PaneBase):
    """Message sent when a worker starts up"""
    hostname: t.Optional[str] = None
    """Hostname worker is running on, if known"""
    backends: t.Optional[t.Dict[str, t.Tuple[str, ...]]] = None
    """Computational backends available to worker, dict from backend -> device"""
    msg: t.Literal['connect'] = 'connect'

class PollMessage(pane.PaneBase):
    """Message polling the server for a job"""
    msg: t.Literal['poll'] = 'poll'

class PingMessage(pane.PaneBase):
    """Message pinging the server, potentially receiving a cancellation"""
    msg: t.Literal['ping'] = 'ping'

class UpdateMessage(pane.PaneBase):
    """Message containing some state update"""
    state: t.Dict[str, t.Any] = pane.field(converter=ReconsStateConverter())
    job_id: JobID
    msg: t.Literal['job_update'] = 'job_update'

class LogMessage(pane.PaneBase):
    """Message corresponding to a log entry"""
    job_id: t.Optional[JobID]

    timestamp: datetime.datetime

    log: str

    logger_name: str
    log_level: int

    line_number: int
    func_name: t.Optional[str] = None
    stack_info: t.Optional[str] = None

    msg: t.Literal['log'] = 'log'

    @classmethod
    def from_logrecord(cls, job_id: t.Optional[JobID], record: logging.LogRecord) -> Self:
        timestamp = datetime.datetime.fromtimestamp(record.created)
        return cls(
            timestamp=timestamp,
            log=record.getMessage(),
            job_id=job_id,
            logger_name=record.name,
            log_level=record.levelno,
            line_number=record.lineno,
            func_name=record.funcName,
            stack_info=record.stack_info,
        )

    def into_record(self, i: int) -> LogRecord:
        return LogRecord.make_unchecked(
            i, self.timestamp, self.log, self.logger_name, self.log_level,
            self.line_number, self.func_name, self.stack_info
        )

class JobResultMessage(pane.PaneBase):
    """Message indicating job has stopped (normally or abnormally)"""
    job_id: JobID
    result: Result
    error: t.Optional[str] = None

    msg: t.Literal['job_result'] = 'job_result'

class WorkerShutdownMessage(pane.PaneBase):
    """Message indicating a worker has been stopped"""
    result: Result
    error: t.Optional[str] = None
    detail: t.Optional[str] = None

    msg: t.Literal['shutdown'] = 'shutdown'


# server -> worker responses

class JobResponse(pane.PaneBase):
    """Response containing a job to run"""
    job_id: JobID
    plan: str
    msg: t.Literal['job'] = 'job'

class OkResponse(pane.PaneBase):
    """Response from server indicating no action should be taken"""
    msg: t.Literal['ok'] = 'ok'

class SignalResponse(pane.PaneBase):
    """Response from server signalling some action (cancellation, shutdown, or reload)"""
    signal: t.Literal['shutdown', 'cancel', 'reload'] = 'cancel'
    urgent: bool = False
    msg: t.Literal['signal'] = 'signal'

WorkerMessage: t.TypeAlias = t.Annotated[t.Union[
    PollMessage, ConnectMessage, PingMessage, UpdateMessage,
    LogMessage, JobResultMessage, WorkerShutdownMessage,
], Tagged('msg')]

ServerResponse: t.TypeAlias = t.Annotated[t.Union[
    JobResponse, SignalResponse, OkResponse
], Tagged('msg')]