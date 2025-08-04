import React from 'react';
import { useAtomValue } from 'jotai';

import { FigureContext, PlotContext, styles } from './plot';
import { Transform1D } from './transform';

const prefixes: Map<number, string> = new Map([
    [-18, "a"],
    [-15, "f"],
    [-12, "p"],
    [-9, "n"],
    [-6, "µ"],
    [-3, "m"],
    [0, ""],
    [3, "k"],
    [6, "M"],
    [9, "G"],
    [12, "T"],
    [15, "P"],
    [18, "E"],
]);

const mantissas: Array<number> = [1, 2, 5];

function divmod(dividend: number, divisor: number): [number, number] {
    const mod = ((dividend % divisor) + divisor) % divisor;
    const quot = (dividend - mod) / divisor;
    return [quot, mod];
}

interface ScalebarProps {
    unit?: string; // = "m"
    unitScale?: number; // = "1"

    minFrac?: number; // = 0.1
    maxFrac?: number; // = 0.5

    radius?: number; // = 4
    height?: number; // = 15
    margin?: number; // = 10
}

export default function Scalebar(props: ScalebarProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot == undefined) {
        throw new Error("Component 'XAxis' must be used inside a 'Plot'");
    }

    const [current, setCurrent] = React.useState<[number, string]>([0, "0 m"]);

    const unit = props.unit ?? "m";
    const unitScale = props.unitScale ?? 1.0;
    const radius = props.radius ?? 4;
    const height = props.height ?? 15;
    const margin = props.margin ?? 10;
    const minFrac = props.minFrac ?? 0.1;
    const maxFrac = props.minFrac ?? 0.5;

    const xtransform = (typeof plot.xaxis === "string") ? useAtomValue(fig.transforms.get(plot.xaxis)!) : new Transform1D();
    const xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    const yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    const fullScale = xaxis.scale;
    const scale = fullScale.applyTransform(xtransform);
    const frameWidth = scale.linDomainSize();

    let [currentSize, currentText] = current;
    let frac = currentSize / frameWidth;
    const force = false;

    if (force || frac < minFrac || frac > maxFrac) {
        const orderOfMagnitude = Math.floor(Math.log10(maxFrac * frameWidth * unitScale));

        const [engOrder, rem] = divmod(orderOfMagnitude, 3);
        const prefix = prefixes.get(3 * engOrder);
        if (prefix === undefined) {
            return <g className={styles["scalebar"]}/>;
        }

        const mult = Math.pow(10, rem);

        for (const mantissa of mantissas) {
            const size = Math.pow(10, orderOfMagnitude) * mantissa / unitScale;
            frac = size / frameWidth;
            if (minFrac <= frac && frac <= maxFrac) {
                const val = mantissa * mult;
                const text = `${val} ${prefix}${unit}`

                setCurrent([size, text]);
                [currentSize, currentText] = [size, text];
            }
        }
    }

    const width = scale.linScale(currentSize);
    const x = scale.rangeFromUnit(1.0) - width - margin;
    const y = yaxis.scale.rangeFromUnit(1.0) - margin - height;
    const textX = scale.rangeFromUnit(1.0) - margin;

    return <g className={styles['scalebar']}>
        <rect rx={radius} height={height} x={x} y={y} width={width} />
        <text x={textX} y={y} dx={-5} dy={-margin} textAnchor="end">{currentText}</text>
    </g>
}