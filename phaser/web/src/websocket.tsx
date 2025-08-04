import { atom, PrimitiveAtom, useStore, createStore } from 'jotai';
import React from 'react';

type Store = ReturnType<typeof createStore>;


class WebsocketConnection {
    socket: WebSocket | null = null;

    public constructor(
        public readonly address: string,
        public readonly store: Store,
        public readonly lastSeen: PrimitiveAtom<Date | null>,
        public readonly status: PrimitiveAtom<string>,
        public readonly onMessage: ((_: MessageEvent<any>) => void) | null,
    ) { }

    connect() {
        this.disconnect();

        console.log(`connecting to '${this.address}'...`);
        this.socket = new WebSocket(this.address);
        this.socket.binaryType = "arraybuffer";

        this.socket.onopen = this._open.bind(this);
        this.socket.onerror = this._error.bind(this);
        this.socket.onclose = this._close.bind(this);
        this.socket.onmessage = this._message.bind(this);
    }

    disconnect() {
        if (this.socket) {
            console.log(`disconnecting from '${this.address}...`);
            this.socket.close();
        }
    }

    private _open(event: Event) {
        this.store.set(this.status, 'Connected');
        this.store.set(this.lastSeen, new Date(event.timeStamp));
    }

    private _error(event: Event) {

    }

    private _close(event: Event) {
        this.store.set(this.status, 'Disconnected');
    }

    private _message(event: MessageEvent<any>) {
        this.store.set(this.lastSeen, new Date(event.timeStamp));

        if (this.onMessage) {
            this.onMessage(event);
        }
    }
}

interface WebsocketProps {
    address: string;
    onMessage: (_: MessageEvent<any>) => void;
}

interface WebsocketReturn {
    status: PrimitiveAtom<string>;
    lastSeen: PrimitiveAtom<Date | null>;
}

export default function websocket(props: WebsocketProps): WebsocketReturn {
    // TODO does this need to be inside a useEffect?
    const store = useStore();

    const status = atom('status');
    const lastSeen = atom<Date | null>(null);

    React.useEffect(() => {
        console.log("Making connection");
        const conn = new WebsocketConnection(
            props.address, store, lastSeen, status, props.onMessage
        );
        conn.connect();

        return () => {
            conn.disconnect();
        }
    }, [store, props.address, props.onMessage])

    return {
        status, lastSeen
    };
}