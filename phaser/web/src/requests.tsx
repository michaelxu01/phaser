import { HTTPError, TimeoutError, ResponsePromise } from 'ky';

type RequestSuccess<T> = {type: 'success', response: Response, body: T};
type RequestError = {type: 'error', response: Response | null, msg: any, error: Error};

// TODO this needs a big cleanup
export async function handleRequest(verb: string, fetchFn: () => ResponsePromise): Promise<RequestSuccess<any> | RequestError> {
    let resp: Response;
    try {
        resp = await fetchFn();
    } catch (error) {
        if (error instanceof HTTPError) {
            const resp = error.response;
            let msg: string;
            let json: any;
            let text: string;
            try {
                text = await resp.text();
                try {
                    msg = processJSONError(JSON.parse(text));
                } catch (e) {
                    msg = text;
                }
            } catch (e) {
                msg = "Couldn't read response text";
            }
            console.error(`HTTP ${resp.status} ${resp.statusText} ${verb}, ${json === undefined ? msg : JSON.stringify(json)}:`, error);
            return {
                type: 'error', response: resp, msg, error
            }
        } else if (error instanceof TimeoutError) {
            console.error(`Request timeout ${verb}: `, error);
            return {
                type: 'error', response: null, msg: "Request timeout", error
            }
        } else {
            console.error(`Unknown error ${verb}: `, error);
            return {
                type: 'error', response: null, msg: "Unknown error", error: error as Error
            }
        }
    }

    let body: any;

    if (resp.headers.get('content-type') == 'application/json') {
        try {
            body = await resp.json();
        } catch (e) {
            const msg = "Invalid JSON in response";
            console.error(`HTTP ${resp.status} ${resp.statusText} ${verb}, ${msg}:`, e);
            return {
                type: 'error', response: resp, msg: "Invalid JSON in response", error: e as Error
            }
        }
    } else {
        try {
            body = await resp.text();
        } catch (e) {
            const msg = "Couldn't read response text";
            console.error(`HTTP ${resp.status} ${resp.statusText} ${verb}, ${msg}:`, e);
            return {
                type: 'error', response: resp, msg: "Couldn't read response body", error: e as Error
            }
        }
    }

    console.log(`HTTP ${resp.status} ${resp.statusText} ${verb}`);
    return {
        type: 'success', response: resp, body
    }
}

/* somewhat equivalent plain fetch method
    const controller = new AbortController();
    setSubmitting();

    setTimeout(() => controller.abort(), 5000);
    try {
        const resp = await fetch(`worker/${worker_type}/start`, {
            method: "POST",
            body: "",
            signal: controller.signal,
        });
        let text: string;
        try {
            text = await resp.text();
        } catch (e) {
            console.error("Error submitting worker (couldn't read response body):", e);
            finishSubmitting();
            return;
        }
        try {
            const json = JSON.parse(text);
            setError(`HTTP ${resp.status} ${resp.statusText}\nJSON: ${JSON.stringify(json)}`);
        } catch (e) {
            setError(`HTTP ${resp.status} ${resp.statusText}\ntext: ${text}`);
        }
    } catch (error) {
        if (error instanceof DOMException && error.name == "AbortError") {
            setError("Request Timeout");
        } else {
            console.error("Error submitting worker: ", error);
            setError("Unknown error");
        }
    }
    finishSubmitting();
*/

function processJSONError(msg: any): string {
    if (msg.result == 'error') {
        return msg.msg;
    }
    return JSON.stringify(msg);
}