import numpy as np
from multiprocessing import Process, Manager
import zmq
import time
import json
import os
import sys
import traceback
from os.path import isfile, join
from pathlib import Path
import queue
from collections import deque
from datetime import datetime
import argparse
from .params import InitParams, InputParams
from .recode_writer import ReCoDeWriter


class rc:
    REQ_TYPE_QUERY = 0
    REQ_TYPE_COMMAND = 1

    FILE_TYPE_BINARY = 0
    FILE_TYPE_MRC = 1
    FILE_TYPE_SEQ = 2
    FILE_TYPE_OTHER = 255

    STATUS_CODE_BUSY = 0  # Processing a request, can't listen but not dead
    STATUS_CODE_AVAILABLE = 1  # Listening
    STATUS_CODE_ERROR = -1  # Dead due to exception
    STATUS_CODE_NOT_READY = -2  # Hasn't stated yet
    STATUS_CODE_IS_CLOSED = -3  # Is closed
    '''
    Expected status flow: -2, 1, 0, 1, 0, 1, 0, -3
    '''
    STATUS_CODES = [STATUS_CODE_BUSY, STATUS_CODE_AVAILABLE, STATUS_CODE_ERROR, STATUS_CODE_NOT_READY,
                    STATUS_CODE_IS_CLOSED]

    MESSAGE_TYPE_INFO_RESPONSE = 0
    MESSAGE_TYPE_ERROR_RESPONSE = -1
    MESSAGE_TYPE_STATUS_RESPONSE = 1
    MESSAGE_TYPE_ACK_RESPONSE = 2
    MESSAGE_TYPE_REQUEST = 3

    MESSAGE_TYPES = [MESSAGE_TYPE_INFO_RESPONSE, MESSAGE_TYPE_ERROR_RESPONSE, MESSAGE_TYPE_STATUS_RESPONSE,
                     MESSAGE_TYPE_ACK_RESPONSE, MESSAGE_TYPE_REQUEST]

    MESSAGE_TYPE_INFORMAL_DESC = {MESSAGE_TYPE_INFO_RESPONSE: 'INFO', MESSAGE_TYPE_ERROR_RESPONSE: 'INFO',
                                  MESSAGE_TYPE_STATUS_RESPONSE: 'STATUS UPDATE',
                                  MESSAGE_TYPE_ACK_RESPONSE: 'ACKNOWLEDGEMENT',
                                  MESSAGE_TYPE_REQUEST: 'REQUEST'}


class MessageData:

    def __init__(self, session_id, message_type, message='', descriptive_message='', mapped_data=None,
                 source_process_id=None, target_process_id=None):

        self._message_payload = {'session_id': session_id}
        if message_type not in rc.MESSAGE_TYPES:
            raise ValueError('Unknown message type')
        self._message_payload['type'] = message_type
        self._message_payload['message'] = message
        self._message_payload['descriptive_message'] = descriptive_message
        self._message_payload['source_process_id'] = source_process_id
        self._message_payload['target_process_id'] = target_process_id

        if mapped_data is None:
            self._message_payload['mapped_data'] = {}
        else:
            self._message_payload['mapped_data'] = mapped_data

    @property
    def session_id(self):
        return self._message_payload['session_id']

    @property
    def type(self):
        return self._message_payload['type']

    @property
    def message(self):
        return self._message_payload['message']

    @property
    def descriptive_message(self):
        return self._message_payload['descriptive_message']

    @property
    def source_process_id(self):
        return self._message_payload['source_process_id']

    @property
    def target_process_id(self):
        return self._message_payload['target_process_id']

    @property
    def mapped_data(self):
        return self._message_payload['mapped_data']

    def set_mapped_data(self, d):
        self._message_payload['mapped_data'] = d

    def add_mapped_data(self, d):
        self._message_payload['mapped_data'].update(d)

    def from_json(self, json_str):
        self._message_payload = json.loads(json_str)
        return self

    def to_json(self):
        return json.dumps(self._message_payload)

    def print(self):
        print(self._message_payload)


class NodeToken(object):

    def __init__(self, node_id, context, ip_address, server_port, publishing_port):
        self._node_id = node_id
        self._context = context
        self._ip_address = ip_address
        self._server_port = server_port
        self._publishing_port = publishing_port

    @property
    def node_id(self):
        return self._node_id

    @property
    def context(self):
        return self._context

    @property
    def ip_address(self):
        return self._ip_address

    @property
    def server_port(self):
        return self._server_port

    @property
    def publishing_port(self):
        return self._publishing_port


class NodeClient:

    def __init__(self, node_token, node_status=rc.STATUS_CODE_NOT_READY):
        self._node_token = node_token
        self._comm_history = []
        self._server_socket = None
        self._sub_socket = None
        self._is_connected = False

    @property
    def token(self):
        return self._node_token

    def connect(self):
        if self._is_connected:
            return
        try:
            self._server_socket = zmq.Context().socket(zmq.REQ)
            self._server_socket.connect(self._node_token.ip_address + ":" + str(self._node_token.server_port))
            self._is_connected = True
        except:
            print('Unable to connect to node')
            self._is_connected = False

    def subscribe(self):
        try:
            self._sub_socket = zmq.Context().socket(zmq.SUB)
            self._sub_socket.connect(self._node_token.ip_address + ":" + str(self._node_token.publishing_port))
        except:
            print('Unable to connect to node')

    def add_to_history(self, message):
        self._comm_history.append(message)

    def send_request(self, md_out):
        self._server_socket.send_json(md_out.to_json())
        s = self._server_socket.recv_json()
        md = MessageData('', 0).from_json(s)
        print('RQM Received:')
        md.print()
        if md.session_id == md_out.session_id:
            if md.mapped_data['req_id'] == md_out.mapped_data['req_id']:
                if md.type == rc.MESSAGE_TYPE_ACK_RESPONSE and md.message == 'ack':
                    return True
        return False

    def close(self):
        if self._is_connected:
            try:
                self._server_socket.close()
                self._is_connected = False
            except Exception as e:
                print(e)


class Logger:

    def __init__(self, token):
        self._node_token = token
        self._socket = None
        # self._logs = {}
        self._logs = deque([])
        self._state = None
        self._session_id = None
        self._verbosity = 0
        self._filename = None
        self._run_name = None

    def start(self, session_id, state, verbosity, run_name, filename=None):
        self._session_id = session_id
        self._state = state
        self._socket = zmq.Context().socket(zmq.PULL)
        self._socket.bind(self._node_token.ip_address + ":" + str(self._node_token.publishing_port))
        self._state[0] = rc.STATUS_CODE_AVAILABLE
        self._verbosity = verbosity
        self._run_name = run_name
        self._filename = filename
        self.run()

    def run(self):
        self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Logger is up and running')
        while True:
            s = self._socket.recv_json()
            md = MessageData('', 0).from_json(s)
            if md.session_id == self._session_id:
                if md.message == 'close':
                    self._shutdown()
                    self._state[0] = rc.STATUS_CODE_IS_CLOSED
                    self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Logger out')
                    return
                self._push(md)

    def _log(self, severity, text, descriptive_text=''):
        m = MessageData(self._session_id, severity, text,
                        descriptive_message=descriptive_text, source_process_id=self._node_token.node_id,
                        mapped_data={'timestamp': datetime.now().strftime('%m/%d/%Y-%H:%M:%S')})
        self._push(m)

    def _push(self, md):
        if 'timestamp' in md.mapped_data:
            timestamp = md.mapped_data['timestamp']
        else:
            timestamp = datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        key = timestamp + '_' + str(md.session_id) + '_' + str(md.source_process_id) + '_' + str(md.type)
        # self._logs[key] = md
        self._logs.append({key: md})
        self._print_msg(timestamp, md)

    def _print_msg(self, timestamp, md):
        formatted_msg = self._format_msg(timestamp, md)
        print(formatted_msg)

    @staticmethod
    def _format_msg(timestamp, md):

        if md.source_process_id is not None and md.source_process_id != '':
            if md.source_process_id == -1:
                src = 'Logger'
            elif md.source_process_id == 8514:
                src = 'Head Node'
            else:
                src = 'Node ' + str(md.source_process_id)
        else:
            src = ''

        typ = rc.MESSAGE_TYPE_INFORMAL_DESC[md.type]

        formatted_msg = '(' + str(typ) + ')' + ' ' + str(timestamp) + ' ' + str(src) + ' : ' + md.message + ' ' + \
                        md.descriptive_message + ' [Session ID:' + str(md.session_id) + ']'

        return formatted_msg

    def _shutdown(self):

        self._socket.close()

        if self._filename is not None and self._filename != '':
            with open(self._filename, "w") as f:
                f.write("\n")
                f.write("Run name: " + self._run_name + " (Session Id: " + str(self._session_id) + ")" + "\n")
                f.write("======================================================\n")
                while len(self._logs) > 0:
                    md = self._logs.popleft()
                    for key in md:
                        timestamp = key[:key.find("_")]
                        f.write(self._format_msg(timestamp, md[key]))
                        f.write("\n")
                        

class ReCoDeServer:

    def __init__(self, mode):

        self._mode = mode

        self._init_params = None
        self._input_params = None
        self._num_threads = None

        self._logger_state = None
        self._node_states = None
        self._node_state_update_timestamps = None
        self._context = None
        self._pub_socket = None
        self._node_token = None
        self._session_id = None

        self._max_attempts = 10
        self._max_non_responsive_time = 15

    def run(self, init_params, input_params=None, dark_data=None, data=None):

        self._init_params = init_params

        # parse and validate input params
        if input_params is None:
            self._input_params = InputParams()
            self._input_params.load(Path(self._init_params.params_filename))
        else:
            self._input_params = input_params

        if self._init_params.mode == 'stream':
            _data = None
        elif self._init_params.mode == 'batch':
            _data = data

        self._num_threads = self._input_params.num_threads

        manager = Manager()
        self._logger_state = manager.dict()
        self._node_states = manager.dict()
        self._node_state_update_timestamps = manager.dict()
        self._context = zmq.Context()

        self._session_id = np.random.randint(10000, 11000)
        nodes = []
        node_clients = []

        # 8514 = HEAD
        self._node_token = NodeToken(8514, self._context, 'tcp://127.0.0.1', 18534, 28534)

        for i in range(self._num_threads):
            token = NodeToken(i, None, 'tcp://127.0.0.1', 18534 + i, 28534)
            node = NodeClient(token)
            node_clients.append(node)
            self._node_states[token.node_id] = rc.STATUS_CODE_NOT_READY
            self._node_state_update_timestamps[token.node_id] = datetime.now()
            rct = ReCoDeNode(token, self._init_params, self._input_params)
            p = Process(target=rct.run,
                        args=(self._session_id, token,
                              self._node_states, self._node_state_update_timestamps, self._logger_state,
                              dark_data, self._input_params, data))
            nodes.append(p)
            p.start()

        time.sleep(0.1)

        # start logger
        token = NodeToken(-1, None, 'tcp://127.0.0.1', -1, 28534)
        logger = Logger(token)
        self._logger_state = {0: rc.STATUS_CODE_NOT_READY}
        logger_process = Process(target=logger.start, args=(self._session_id, self._logger_state,
                                                            self._init_params.verbosity, self._init_params.run_name,
                                                            self._init_params.log_filename))
        logger_process.start()

        time.sleep(0.1)

        # connect to logger
        self._pub_socket = self._context.socket(zmq.PUSH)
        self._pub_socket.connect(self._node_token.ip_address + ":" + str(self._node_token.publishing_port))
        self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Welcome to this session')

        # connect to RC nodes
        for i in range(self._num_threads):
            node_clients[i].connect()

        self._broadcast(node_clients, self._session_id, 1, 'start')

        if self._mode == 'batch':
            self._broadcast(node_clients, self._session_id, 2, 'process_file')
        elif self._mode == 'stream':
            self._recode_queue_manager(self._init_params.directory_path, self._init_params.max_count,
                                       self._init_params.chunk_time_in_sec, node_clients)

        # close RC nodes
        self._broadcast(node_clients, self._session_id, 3, 'close')
        for i in range(self._num_threads):
            node_clients[i].close()
            nodes[i].join()

        # close logger
        self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'close')
        logger_process.join()

    def _spawn_replacement_node(self, node_id):
        return

    def _broadcast(self, nodes, session_id, req_id, msg_text):

        _node_acknowledged = np.zeros((len(nodes)), dtype=np.bool)
        _node_attempts = np.zeros((len(nodes)), dtype=np.int)

        n_available_nodes = 0
        for node in nodes:
            if self._node_states[node.token.node_id] != rc.STATUS_CODE_ERROR:
                n_available_nodes += 1

        while np.min(_node_attempts) < self._max_attempts and n_available_nodes > 0:
            for i, node in enumerate(nodes):
                # if rct thread i has not already acknowledged
                if self._node_states[node.token.node_id] != rc.STATUS_CODE_ERROR and not _node_acknowledged[i]:
                    # try (re)sending a message
                    m = MessageData(session_id, rc.MESSAGE_TYPE_REQUEST, msg_text, mapped_data={'req_id': req_id},
                                    source_process_id=self._node_token.node_id, target_process_id=node.token.node_id)
                    if self._node_states[node.token.node_id] == rc.STATUS_CODE_AVAILABLE:
                        _node_acknowledged[i] = node.send_request(m)
                        print('RQM sent to server', i, '(port', node.token.server_port, '):')
                        m.print()
                        if not _node_acknowledged[i]:
                            if _node_attempts[i] > self._max_attempts:
                                node.set_status(rc.STATUS_CODE_ERROR)
                            else:
                                _node_attempts[i] += 1
                    else:
                        lct = self._node_state_update_timestamps[node.token.node_id]
                        non_responsive_time = (datetime.now() - lct).total_seconds()
                        if non_responsive_time > self._max_non_responsive_time:
                            print('Head Node: Server', i,
                                  'is non-responsive. (Last contact time:', lct, '). Starting replacement node...')
                            self._spawn_replacement_node(i)
                        else:
                            print('Head Node: Server', i, 'is busy. (Last contact time:', lct, '). Continuing...')

            # check if all nodes have acknowledged
            if np.sum(_node_acknowledged) == len(nodes):
                return

            n_available_nodes = 0
            for node in nodes:
                if self._node_states[node.token.node_id] != rc.STATUS_CODE_ERROR:
                    n_available_nodes += 1

            time.sleep(1)

        return _node_acknowledged, n_available_nodes

    def _log(self, severity, text, descriptive_text=''):
        m = MessageData(self._session_id, severity, text,
                        descriptive_message=descriptive_text, source_process_id=self._node_token.node_id,
                        mapped_data={'timestamp': datetime.now().strftime('%m/%d/%Y-%H:%M:%S')}).to_json()
        self._pub_socket.send_json(m)

    def _recode_queue_manager(self, source_dir, max_count, chunk_time_in_sec, node_clients):

        if not os.path.isdir(source_dir):
            raise ValueError('Directory ' + source_dir + ' not found.')

        for f in os.listdir(source_dir):
            os.remove(os.path.join(source_dir, f))

        q = queue.Queue()
        queued_files = {}
        is_first = True

        count = 0

        has_interrupt = False
        while not has_interrupt and count < max_count - 1:

            f_list = [f for f in os.listdir(source_dir) if isfile(join(source_dir, f)) and f.endswith(".seq")]

            has_new_file = False
            for f in f_list:
                if not f in queued_files:
                    q.put(f)
                    queued_files[f] = 1
                    has_new_file = True

            if has_new_file and q.qsize() > 1:

                fname = q.get()

                if is_first:
                    is_first = False
                    continue

                # Rename
                new_fname = join(source_dir, "Next_Stream.seq")
                os.rename(join(source_dir, fname), new_fname)

                # Read and Process
                print("Monitor Thread: Processing " + fname)
                start_time = time.time()
                st = datetime.now()
                self._broadcast(node_clients, self._session_id, 2, 'process_file')

                # Wait for all threads to finish or max timeout to be reached
                all_nodes_free = False
                wait_time = (datetime.now() - st).total_seconds()
                while all_nodes_free is False and wait_time < self._max_non_responsive_time:
                    time.sleep(0.1)
                    all_nodes_free = True
                    for i, node in enumerate(node_clients):
                        if self._node_states[node.token.node_id] != rc.STATUS_CODE_AVAILABLE:
                            all_nodes_free = False
                            break
                    wait_time = (datetime.now() - st).total_seconds()

                elapsed_time = time.time() - start_time
                print("Monitor Thread: Processed in " + str(elapsed_time) + " seconds.")
                count += 1

                # Delete
                print("Monitor Thread: Clearing from " + source_dir)
                start_time = time.time()
                os.remove(join(source_dir, "Next_Stream.seq"))
                elapsed_time = time.time() - start_time
                print("Monitor Thread: Cleared in " + str(elapsed_time) + " seconds.\n")

        # print(f_list)
        # print('{0:d} files in Queue.'.format(q.qsize()))

        # Process the last chunk
        fname = q.get()

        # Wait for the last file to be written (necessary when processing is faster than acquisition)
        time.sleep(chunk_time_in_sec + 1)

        # Rename
        new_fname = join(source_dir, "Next_Stream.seq")
        os.rename(join(source_dir, fname), new_fname)

        # Read and Process
        print("Monitor Thread: Processing " + fname)
        start_time = time.time()
        st = datetime.now()
        self._broadcast(node_clients, self._session_id, 2, 'process_file')

        # Wait for all threads to finish or max timeout to be reached
        all_nodes_free = False
        wait_time = (datetime.now() - st).total_seconds()
        while all_nodes_free is False and wait_time < self._max_non_responsive_time:
            time.sleep(0.1)
            all_nodes_free = True
            for i, node in enumerate(node_clients):
                if self._node_states[node.token.node_id] != rc.STATUS_CODE_AVAILABLE:
                    all_nodes_free = False
                    break
            wait_time = (datetime.now() - st).total_seconds()

        elapsed_time = time.time() - start_time
        print("Monitor Thread: Processed in " + str(elapsed_time) + " seconds.")
        count += 1


class ReCoDeNode:

    def __init__(self, node_token, init_params, input_params):
        self._node_token = node_token
        self._pid = node_token.node_id
        self._sport = node_token.server_port
        self._pub_port = node_token.publishing_port
        self._init_params = init_params
        self._input_params = input_params

        self._socket = None
        self._pub_socket = None
        self._node_states = None
        self._node_state_update_timestamps = None
        self._logger_state = None
        self._session_id = None

        self._frames = []
        self._timestamps = []

        self._recode_writer = None

    def _set_state(self, state):
        self._node_states[self._pid] = state
        self._node_state_update_timestamps[self._pid] = datetime.now()

    def _log(self, severity, text, descriptive_text=''):
        m = MessageData(self._session_id, severity, text,
                        descriptive_message=descriptive_text, source_process_id=self._pid,
                        mapped_data={'timestamp': datetime.now().strftime('%m/%d/%Y-%H:%M:%S')}).to_json()
        self._pub_socket.send_json(m)

    def run(self, session_id, token, node_states, node_state_update_timestamps, logger_state,
            dark_data=None, input_params=None, data=None):

        self._node_token = token
        self._node_states = node_states
        self._node_state_update_timestamps = node_state_update_timestamps
        self._logger_state = logger_state
        self._session_id = session_id

        # start server
        try:
            context = zmq.Context()

            self._socket = context.socket(zmq.REP)
            self._socket.bind(self._node_token.ip_address + ":" + str(self._sport))

            self._pub_socket = context.socket(zmq.PUSH)
            self._pub_socket.connect(self._node_token.ip_address + ":" + str(self._pub_port))

            self._set_state(rc.STATUS_CODE_AVAILABLE)
            print('Node', self._pid, 'Ready at port', self._sport)

        except Exception as e:
            self._set_state(rc.STATUS_CODE_ERROR)
            print('Node:', self._node_token.node_id, '=>', 'Exception:', e)
            self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, "Exception",
                      descriptive_text=traceback.format_exc())
            traceback.print_tb(e.__traceback__)
            return

        while True:
            self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Node ' + str(self._pid) + ' Waiting...')
            s = self._socket.recv_json()
            md = MessageData('', 0).from_json(s)
            print('Node', self._pid, ' Received:')
            md.print()
            if md.session_id == session_id:
                if md.type == rc.MESSAGE_TYPE_REQUEST:

                    if md.message == 'start':
                        self._send_ack(md)
                        self._set_state(rc.STATUS_CODE_BUSY)
                        try:
                            self._open(dark_data=dark_data, input_params=input_params)
                            self._start()
                            self._set_state(rc.STATUS_CODE_AVAILABLE)
                        except Exception as e:
                            self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, "Exception",
                                      descriptive_text=traceback.format_exc())
                            traceback.print_tb(e.__traceback__)
                            self._set_state(rc.STATUS_CODE_ERROR)

                    elif md.message == 'process_file':
                        self._send_ack(md)
                        self._set_state(rc.STATUS_CODE_BUSY)
                        try:
                            self._process_file(data=data)
                            self._set_state(rc.STATUS_CODE_AVAILABLE)
                        except Exception as e:
                            self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, "Exception",
                                      descriptive_text=traceback.format_exc())
                            traceback.print_tb(e.__traceback__)
                            self._set_state(rc.STATUS_CODE_ERROR)

                    elif md.message == 'close':
                        self._send_ack(md)
                        try:
                            self._close(node_states)
                            self._set_state(rc.STATUS_CODE_IS_CLOSED)
                            return
                        except Exception as e:
                            self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, "Exception",
                                      descriptive_text=traceback.format_exc())
                            traceback.print_tb(e.__traceback__)
                            self._set_state(rc.STATUS_CODE_ERROR)
                            return

                    else:
                        print('Unknown request')
                        self._set_state(rc.STATUS_CODE_ERROR)

    def _send_ack(self, md):
        m = MessageData(
            md.session_id, rc.MESSAGE_TYPE_ACK_RESPONSE, 'ack', mapped_data={'req_id': md.mapped_data['req_id']},
            source_process_id=self._pid, target_process_id=0
        ).to_json()
        self._socket.send_json(m)

    def _open(self, dark_data=None, input_params=None):
        # create ReCoDeWriter depending on mode, if mode is stream image_filename is path_to_ramdisk\Next_Stream.seq
        # load calibration data
        if self._init_params.mode == "stream":
            _image_filename = os.path.join(self._init_params.directory_path, "Next_Stream.seq")
        elif self._init_params.mode == "batch":
            _image_filename = self._init_params.image_filename

        self._recode_writer = ReCoDeWriter(image_filename=_image_filename,
                                           dark_data=dark_data,
                                           dark_filename=self._init_params.calibration_filename,
                                           output_directory=self._init_params.output_directory,
                                           input_params=input_params,
                                           params_filename=self._init_params.params_filename,
                                           mode=self._init_params.mode,
                                           validation_frame_gap=self._init_params.validation_frame_gap,
                                           log_filename=self._init_params.log_filename,
                                           run_name=self._init_params.run_name,
                                           verbosity=self._init_params.verbosity,
                                           use_c=self._init_params.use_c,
                                           max_count=self._init_params.max_count,
                                           chunk_time_in_sec=self._init_params.chunk_time_in_sec,
                                           node_id=self._pid,
                                           buffer_size_in_frames=10)

    def _start(self):
        # create output rc part file
        # serialize header(s) to part file
        # allocate internal buffers
        self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Processing...')
        self._recode_writer.start()

    def _process_file(self, data=None):
        # read source frames
        # reduce-compress and serialize to destination
        run_metrics = self._recode_writer.run(data=data)
        self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Processed ' + str(run_metrics['run_frames']) + ' frames')
        if 'run_dose_rates' in run_metrics:
            self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Estimated dose rate = ' + str(run_metrics['run_dose_rates'][-1]))

        # at the end of each run, update q with the frame_ids processed
        return

    def _close(self, node_states):
        # update recode header with true frame count, flush remaining data and close file
        self._recode_writer.close()
        self._socket.close()
        self._log(rc.MESSAGE_TYPE_INFO_RESPONSE, 'Clean shutdown')
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ReCoDe Server')
    parser.add_argument('--source', dest='source', action='store', default='',
                        help='if mode is batch then file to be processed, otherwise path to folder containing data '
                             '(typically inside RAM disk for on-the-fly)')
    parser.add_argument('--calibration_file', dest='calibration_file', action='store', default='',
                        help='path to calibration file')
    parser.add_argument('--out_dir', dest='out_dir', action='store', default='', help='output directory')
    parser.add_argument('--params_file', dest='params_file', action='store', default='', help='path to params file')
    parser.add_argument('--mode', dest='mode', action='store', default='batch', help='batch or stream')
    parser.add_argument('--validation_frame_gap', dest='validation_frame_gap', action='store', type=int, default=-1,
                        help='validation frame gap')
    parser.add_argument('--log_file', dest='log_file', action='store', default='', help='path to log file')
    parser.add_argument('--run_name', dest='run_name', action='store', default='run_1', help='run name')
    parser.add_argument('--verbosity', dest='verbosity', action='store', type=int, default=0, help='verbosity level')
    parser.add_argument('--use_c', dest='use_c', action='store_true', help='')
    parser.add_argument('--max_count', dest='max_count', action='store', type=int, default=1,
                        help='the number of chunks to process')
    parser.add_argument('--chunk_time_in_sec', dest='chunk_time_in_sec', action='store', type=int, default=1,
                        help='seconds of data contained in each chunk')

    args = parser.parse_args()

    init_params = InitParams(args.mode, args.out_dir, image_filename=args.source, directory_path=args.source,
                             calibration_file=args.calibration_file, params_filename=args.params_file,
                             validation_frame_gap=args.validation_frame_gap, log_filename=args.log_file,
                             run_name=args.run_name, verbosity=args.verbosity, use_c=False, max_count=args.max_count,
                             chunk_time_in_sec=args.chunk_time_in_sec)

    if not init_params.validate():
        raise ValueError('Invalid initialization parameters')

    server = ReCoDeServer(args.mode)
    server.run(init_params)
