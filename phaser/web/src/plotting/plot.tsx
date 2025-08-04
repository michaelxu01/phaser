import React, { useMemo } from 'react';
import { atom, useAtom, useAtomValue, Atom, PrimitiveAtom } from 'jotai';

import * as d3_format from 'd3-format';
import { NArray } from 'wasm-array';
import { np } from '../wasm-array';
import { useElementSize } from '@mantine/hooks';

import { Transform1D, Transform2D } from './transform';
import { PlotScale, Pair, isClose } from './scale';
import { Zoomer } from "./zoom";
import * as styles from "./plots.module.css";
export { styles };

export interface AxisSpec {
    scale: PlotScale

    translateExtent?: Pair | boolean
    label?: string
    labelOffset?: number
    show?: boolean | 'one'

    ticks?: number
    tickFormat?: string
    tickLength?: number
}

export interface Axis {
    scale: PlotScale

    translateExtent: Pair
    label?: string
    labelOffset?: number
    show: boolean | 'one'

    ticks?: number
    tickFormat?: string
    tickLength?: number
}

function normalize_axis(axis: AxisSpec | PlotScale): Axis {
    if (axis instanceof PlotScale) {
        axis = {
            scale: axis
        };
    }

    axis.show = ("show" in axis) ? axis.show : true;

    if (axis.translateExtent === true || !("translateExtent" in axis)) {
        axis.translateExtent = axis.scale.domain;
    } else if (!axis.translateExtent) {
        axis.translateExtent = [-Infinity, Infinity];
    }

    return axis as Axis;
}

export interface ColorScale {
    cmap?: string
    range?: [number, number]
    label?: string
}

export interface FigureContextData<K> {
    axes: Map<K, Axis>
    transforms: Map<K, PrimitiveAtom<Transform1D>>

    scales: Map<K, ColorScale>
    currentRanges: Map<K, PrimitiveAtom<[number, number] | null>>

    zoomExtent: Pair
}

export const FigureContext = React.createContext<FigureContextData<string> | undefined>(undefined);

interface FigureProps {
    axes: Map<string, AxisSpec | PlotScale>
    zoomExtent?: Pair

    scales?: Map<string, ColorScale>

    children?: React.ReactNode
}

function mapValues<K, V, T>(map: Map<K, V>, func: (value: V) => T): Map<K, T> {
    return new Map([...map].map(([k, v]) => [k, func(v)]));
}

export function Figure({
    axes: inputAxes,
    scales = new Map(),
    zoomExtent,
    children
}: FigureProps) {

    const axes = useMemo(() => mapValues(inputAxes, normalize_axis), [inputAxes]);
    const transforms = useMemo(() => mapValues(axes, () => atom(new Transform1D())), [axes]);

    const currentRanges = useMemo(() => mapValues(scales, v => atom(v.range ?? null)), [scales]);

    return <FigureContext.Provider value={{
        axes,
        transforms,
        zoomExtent: zoomExtent || [1, Infinity],
        scales,
        currentRanges,
    }}>
        {children}
    </FigureContext.Provider>;
}

export interface PlotContextData<K> {
    xaxis: K | Axis
    yaxis: K | Axis

    xaxis_pos: 'bottom' | 'top'
    yaxis_pos: 'left' | 'right'

    fixedAspect: boolean
    clipId: string
}

export const PlotContext = React.createContext<PlotContextData<string> | undefined>(undefined);

export function makeId(prefix: string): string {
    return prefix + `-${d3_format.format("06g")(Math.floor(Math.random() * 1000000))}`;
}

interface AxisProps {
    label?: string | undefined
}

export function XAxis(props: AxisProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot == undefined) {
        throw new Error("Component 'XAxis' must be used inside a 'Plot'");
    }

    let xtransform = (typeof plot.xaxis === "string") ? useAtomValue(fig.transforms.get(plot.xaxis)!) : new Transform1D();
    let xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    let yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    let cross_pos = (plot.xaxis_pos == "top") ? 0.0 : 1.0;
    let sign = (plot.xaxis_pos == "top") ? -1.0 : 1.0;
    const className = (plot.xaxis_pos == "top") ? styles['top-axis'] : styles['bot-axis'];

    let fullScale = xaxis.scale;
    let scale = fullScale.applyTransform(xtransform);

    const labelOffset = xaxis.labelOffset ?? 50;

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className={styles["axis-label"]} transform={`translate(${scale.rangeFromUnit(0.5)}, ${sign * labelOffset})`}>
            {props.label}
        </text>;
    }

    // TODO factor some stuff out
    // TODO replace with path

    const fmt = d3_format.format(xaxis.tickFormat ?? "~g");
    const tickLength = xaxis.tickLength ?? 8;

    let ticks = scale.ticks(xaxis.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = scale.transform(val);
        return <g className={styles["tick"]} key={val}>
            <line x1={pos} x2={pos} y1={0} y2={sign * tickLength} stroke="inherit"/>
            <text x={pos} y={sign * tickLength} dy={`${sign*0.9}em`}>{text}</text>
        </g>;
    });

    let ax_ypos = yaxis.scale.rangeFromUnit(cross_pos);
    let [ax_start, ax_stop] = scale.range;
    return <g className={className} transform={`translate(0, ${ax_ypos})`}>
        <line x1={ax_start} x2={ax_stop} y1="0" y2="0" stroke="inherit"/>
        { ticks }
        { label }
    </g>;
}

export function YAxis(props: AxisProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot === undefined) {
        throw new Error("Component 'YAxis' must be used inside a 'Plot'");
    }

    let ytransform = (typeof plot.yaxis === "string") ? useAtomValue(fig.transforms.get(plot.yaxis)!) : new Transform1D();
    let xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    let yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    let cross_pos = (plot.yaxis_pos == "left") ? 0.0 : 1.0;
    let sign = (plot.yaxis_pos == "left") ? -1.0 : 1.0;
    const className = (plot.yaxis_pos == "left") ? styles['left-axis'] : styles['right-axis'];

    let fullScale = yaxis.scale;
    let scale = fullScale.applyTransform(ytransform);

    const labelOffset = yaxis.labelOffset ?? 90;

    let label: React.ReactElement | undefined = undefined;
    if (props.label) {
        label = <text className={styles["axis-label"]} transform={`translate(${sign * labelOffset}, ${scale.rangeFromUnit(0.5)}) rotate(${sign * -90})`}>
            {props.label}
        </text>;
    }

    const fmt = d3_format.format(yaxis.tickFormat ?? "~g");
    const tickLength = yaxis.tickLength ?? 8;

    let ticks = scale.ticks(yaxis.ticks ?? 4).map((val) => {
        const text = fmt(val);
        const pos = scale.transform(val);
        return <g className={styles["tick"]} key={val}>
            <line x1={sign * tickLength} x2={0} y1={pos} y2={pos} stroke="inherit"/>
            <text x={sign * tickLength} y={pos} dx={`${sign*0.3}em`} dy="0.4em">{text}</text>
        </g>;
    });

    let ax_xpos = xaxis.scale.rangeFromUnit(cross_pos);
    let [ax_start, ax_stop] = scale.range;
    return <g className={className} transform={`translate(${ax_xpos}, 0)`}>
        <line x1="0" x2="0" y1={ax_start} y2={ax_stop} stroke="inherit"/>
        { ticks }
        { label }
    </g>;
}

interface PlotProps {
    xaxis?: string | AxisSpec
    yaxis?: string | AxisSpec

    fixedAspect?: boolean /* = false*/
    /*width: number
    height: number
    xDomain?: [number, number]
    yDomain?: [number, number]*/
    margins?: [number, number, number, number]

    show_xaxis?: boolean
    show_yaxis?: boolean

    xaxis_pos?: 'bottom' | 'top'
    yaxis_pos?: 'left' | 'right'

    children?: React.ReactNode
}

export const Plot = React.memo(function Plot (props: PlotProps) {
    //console.log("Redrawing Plot");

    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'Plot' must be used inside a 'Figure'");
    }

    if (!props.xaxis || !props.yaxis) {
        throw new Error("Component 'Plot' must have xaxis and yaxis props defined.");
    }

    let xaxis = (typeof props.xaxis === "string") ? fig.axes.get(props.xaxis)! : normalize_axis(props.xaxis);
    let yaxis = (typeof props.yaxis === "string") ? fig.axes.get(props.yaxis)! : normalize_axis(props.yaxis);
    if (!xaxis) throw new Error("Invalid xaxis passed to component 'Plot'");
    if (!yaxis) throw new Error("Invalid yaxis passed to component 'Plot'");

    const xaxis_pos = props.xaxis_pos ?? 'bottom';
    const yaxis_pos = props.yaxis_pos ?? 'left';

    const show_xaxis = props.show_xaxis ?? !!xaxis.show;
    const show_yaxis = props.show_yaxis ?? !!yaxis.show;

    let clippedChildren: React.ReactNode[] = [];
    let children: React.ReactNode[] = [];

    React.Children.forEach(props.children, child => {
        // TODO this is a huge hack
        if (child && typeof child === 'object' && 'type' in child) {
            if (typeof child.type === 'function' && child.type.name == "Scalebar") {
                children.push(child);
                return;
            }
        }
        clippedChildren.push(child);
    });

    if (show_xaxis) children.push(<XAxis label={xaxis.label} key="xaxis"/>)
    if (show_yaxis) children.push(<YAxis label={yaxis.label} key="yaxis"/>)

    const dims = calc_plot_dims(fig, xaxis, yaxis, show_xaxis, show_yaxis, xaxis_pos, yaxis_pos, props.margins);

    const clipId = React.useMemo(() => makeId("ax-clip"), []);

    let ctx: PlotContextData<string> = {
        xaxis: (typeof props.xaxis === "string") ? props.xaxis : xaxis,
        yaxis: (typeof props.yaxis === "string") ? props.yaxis : yaxis,
        fixedAspect: props.fixedAspect ?? false,

        xaxis_pos: xaxis_pos,
        yaxis_pos: yaxis_pos,
        clipId: clipId,
    };

    return <PlotContext.Provider value={ctx}> <Zoomer>
        <svg className={styles["plot"]} viewBox={dims.viewBox.join(" ")} width={dims.totalWidth} height={dims.totalHeight}>
            <clipPath id={clipId}><rect x={0} y={0} width={dims.width} height={dims.height}/></clipPath>
            <g className={styles["axis-cont"]}>
                <rect className={styles["axis-box"]} width={dims.width} height={dims.height}/>
                <g className={styles["axis-clip"]} clipPath={`url(#${clipId})`}>
                    <g className={styles["zoom"]}>
                        { clippedChildren }
                    </g>
                </g>
                { children }
            </g>
        </svg>
    </Zoomer> </PlotContext.Provider>;
});

const isPlot = (val: any): val is React.ReactElement<PlotProps, typeof Plot> =>
    React.isValidElement(val) && typeof val.type['type'] == "function" && val.type['type'].name == "Plot";

interface PlotDims {
    width: number
    height: number
    totalWidth: number
    totalHeight: number
    viewBox: [number, number, number, number]
}

function calc_axis_size(
    axis: Axis,
    pos: 'bottom' | 'top' | 'left' | 'right',
): number {
    // todo be smarter here
    if (['bottom', 'top'].includes(pos)) {
        return 60;
    }
    return 150;
}

function calc_plot_dims(
    fig: FigureContextData<string>,
    xaxis: Axis, yaxis: Axis,
    show_xaxis: boolean, show_yaxis: boolean,
    xaxis_pos: 'bottom' | 'top', yaxis_pos: 'left' | 'right',
    margins?: [number, number, number, number]
): PlotDims {
    let [xscale, yscale] = [xaxis.scale, yaxis.scale] ;

    const [width, height] = [xscale.rangeSize(), yscale.rangeSize()];

    let marginTop: number, marginRight: number, marginBottom: number, marginLeft: number;

    if (margins) {
        [marginTop, marginRight, marginBottom, marginLeft] = margins;
    } else {
        [marginTop, marginRight, marginBottom, marginLeft] = [15, 15, 15, 15];
        if (show_xaxis) {
            const axis_size = calc_axis_size(xaxis, xaxis_pos);
            if (xaxis_pos == 'bottom')
                marginBottom += axis_size;
            else
                marginTop += axis_size;
        }
        if (show_yaxis) {
            const axis_size = calc_axis_size(yaxis, yaxis_pos);
            if (yaxis_pos == 'left')
                marginLeft += axis_size;
            else
                marginRight += axis_size;
        }
    }

    const totalWidth = width + marginLeft + marginRight;
    const totalHeight = height + marginBottom + marginTop; 
    const viewBox: [number, number, number, number] = [-marginLeft, -marginTop, totalWidth, totalHeight];

    return {
        width: width, height: height,
        totalWidth: totalWidth, totalHeight: totalHeight,
        viewBox: viewBox,
    }
}

interface PlotGridProps {
    ncols: number;
    nrows: number;

    xaxes: string | ReadonlyArray<string>;
    yaxes: string | ReadonlyArray<string>;

    pad?: string | number; /* = 0px */

    zoomExtent?: [number, number];

    children?: React.ReactNode
}

const PlotGrid = React.memo(function PlotGrid(props: PlotGridProps) {
    const [ncols, nrows] = [props.ncols, props.nrows];

    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'PlotGrid' must be used inside a 'Figure'");
    }

    let xaxes: Array<string>;
    if (typeof(props.xaxes) === "string") {
        // share x axis
        xaxes = Array(props.ncols).fill(props.xaxes);
    } else {
        if (props.xaxes.length != ncols) {
            throw new Error("PlotGrid: `xaxes` must an axis key or an array of `ncols` axis keys");
        }
        xaxes = [...props.xaxes];
    }

    let yaxes: Array<string>;
    if (typeof(props.yaxes) === "string") {
        // share y axis
        yaxes = Array(props.nrows).fill(props.yaxes);
    } else {
        if (props.yaxes.length != nrows) {
            throw new Error("PlotGrid: `yaxes` must an axis key or an array of `nrows` axis keys");
        }
        yaxes = [...props.yaxes];
    }

    if (React.Children.count(props.children) > nrows * ncols) {
        throw new Error(`PlotGrid: Too many children, maximum is nrows*ncols = ${nrows * ncols}`);
    }

    let widths: Array<number> = Array(props.ncols).fill(0);
    let heights: Array<number> = Array(props.nrows).fill(0);

    const children = React.Children.map(props.children, (child, i) => {
        const [row, col] = [Math.floor(i / ncols), i % ncols];

        if (isPlot(child)) {
            const child_props = child.props as PlotProps;

            const props_xaxis = child_props.xaxis ?? xaxes[col];
            const props_yaxis = child_props.yaxis ?? yaxes[row];
            let xaxis = (typeof props_xaxis === "string") ? fig.axes.get(props_xaxis)! : normalize_axis(props_xaxis);
            let yaxis = (typeof props_yaxis === "string") ? fig.axes.get(props_yaxis)! : normalize_axis(props_yaxis);

            const xaxis_pos = child_props.xaxis_pos ?? 'bottom';
            const yaxis_pos = child_props.yaxis_pos ?? 'left';

            const show_xaxis: boolean = child_props.show_xaxis ?? (
                xaxis.show == "one" ? row == (xaxis_pos == 'top' ? 0 : nrows - 1) : xaxis.show
            );
            const show_yaxis: boolean = child_props.show_yaxis ?? (
                yaxis.show == "one" ? col == (yaxis_pos == 'left' ? 0 : ncols - 1) : yaxis.show
            );

            const dims = calc_plot_dims(fig, xaxis, yaxis, show_xaxis, show_yaxis, xaxis_pos, yaxis_pos, child.props.margins);
            widths[col] = Math.max(widths[col], dims.totalWidth);
            heights[row] = Math.max(heights[row], dims.totalHeight);

            const plotProps = {xaxis: props_xaxis, yaxis: props_yaxis, show_xaxis: show_xaxis, show_yaxis: show_yaxis};
            child = React.cloneElement(child, plotProps);
        }

        const style = {
            gridColumn: col + 1,
            gridRow: row + 1,
        };

        return <div className="plotGridItem" style={style}> { child } </div>
    });

    const gridStyle = {
        display: "grid",
        gridTemplateColumns: widths.map((v) => `${v}px`).join(' '),
        gridTemplateRows: heights.map((v) => `${v}px`).join(' '),
        gap: props.pad ?? "0px",
    };

    return <div className="plotGrid" style={gridStyle}>
        { children }
    </div>;
})

interface PlotListProps {
    xaxis: string;
    yaxis: string;

    pad?: number; /* = 0px */
    maxWidth?: number;

    zoomExtent?: [number, number];

    children?: React.ReactNode
}

export function PlotList(props: PlotListProps) {
    const fig = React.useContext(FigureContext);
    if (fig === undefined) {
        throw new Error("Component 'PlotList' must be used inside a 'Figure'");
    }

    const pad = props.pad ?? 0;

    const { ref, width = 1 } = props.maxWidth ?
        { ref: undefined, width: props.maxWidth } : useElementSize<HTMLDivElement>();

    const xaxis = fig.axes.get(props.xaxis)!;
    const yaxis = fig.axes.get(props.yaxis)!;

    const child_plots = React.Children.toArray(props.children).filter(isPlot);
    const n_plots = child_plots.length;
    const child_props = child_plots.map((child) => child.props);

    let n_cols = 0;
    let space = width;

    for (let i = 0; i < n_plots; i++) {
        const xaxis_pos = child_props[i].xaxis_pos ?? 'bottom';
        const yaxis_pos = child_props[i].yaxis_pos ?? 'left';
        const show_yaxis = child_props[i].show_yaxis ?? (
            yaxis.show == "one" ? i == (yaxis_pos == 'left' ? 0 : -1) : yaxis.show
        );

        const plotWidth = calc_plot_dims(fig, xaxis, yaxis, false, show_yaxis, xaxis_pos, yaxis_pos).totalWidth;

        space -= plotWidth + (i > 0 ? pad : 0);

        if (space < 0) break;
        n_cols += 1;
    }

    n_cols = Math.max(n_cols, 1);
    const n_rows = Math.ceil(n_plots / n_cols);

    return <div ref={ref as React.RefObject<HTMLDivElement> | undefined}>
        <PlotGrid ncols={n_cols} nrows={n_rows} xaxes={props.xaxis} yaxes={props.yaxis} pad={props.pad} zoomExtent={props.zoomExtent}>{props.children}</PlotGrid>
    </div>
}

interface PlotImageProps {
    data: NArray | null
    scale: string

    xlim?: [number, number] // [min, max]
    ylim?: [number, number] // [min, max]
}

export const PlotImage = React.memo(function (props: PlotImageProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot === undefined) {
        throw new Error("Component 'PlotImage' must be used inside a 'Plot'");
    }

    let xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    let yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    const scale = fig.scales.get(props.scale);
    if (!scale) throw new Error(`Component 'PlotImage' passed invalid scale '${scale}'`);

    const data = props.data;
    if (!data) return null;

    const [currentRange, setCurrentRange] = useAtom(fig.currentRanges.get(props.scale)!);

    const [height, width] = data.shape.values();

    const xlim = xaxis.scale.transform(props.xlim ?? xaxis.scale.domain);
    const ylim = yaxis.scale.transform(props.ylim ?? yaxis.scale.domain);

    let transform = Transform2D.fromBounds([0, width], [0, height]).compose(
        Transform2D.fromBounds(xlim, ylim).invert()
    );

    /*let transform = Transform2D.fromBounds(xlim, ylim)
        .scale(Math.abs(xlim[1] - xlim[0]) / width, Math.abs(ylim[1] - ylim[0]) / height);*/
    //let transform = new Transform2D();

    const canvasRef: React.MutableRefObject<HTMLCanvasElement | null> = React.useRef(null);

    React.useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !np) return;

        let range;
        if (!scale.range) {
            range = [np.nanmin(data), np.nanmax(data)];
            setCurrentRange(range);
        } else {
            range = currentRange;
        }

        const ctx = canvas.getContext('2d')!;
        const imageData = ctx.createImageData(width, height);

        imageData.data.set(
            np.expr`(${data} - ${range[0]}) / (${range[1]} - ${range[0]})`.apply_cmap(scale.cmap ?? 'magma')
        );

        ctx.putImageData(imageData, 0, 0);
    }, [data, currentRange]);

    // fix for safari rendering the canvas outside of the SVG context
    // this ensures that the child contents render with the correct transformation, with correct clip paths
    if (typeof navigator !== 'undefined' && /Version\/[\d\.]+.*Safari/.test(navigator.userAgent)) { // is safari
        let xtrans = (typeof plot.xaxis === "string") ? useAtom(fig.transforms.get(plot.xaxis)!) : new Transform1D();
        let ytrans = (typeof plot.xaxis === "string") ? useAtom(fig.transforms.get(plot.xaxis)!) : new Transform1D();

        React.useEffect(() => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const parent = canvas.parentNode as SVGForeignObjectElement;
            const t = parent.getCTM()!;
            const viewbox = parent.ownerSVGElement!.viewBox.animVal;

            // we have to apply the SVG transformation stack manually
            canvas.style.transform = `matrix(${t.a}, ${t.b}, ${t.c}, ${t.d}, ${t.e}, ${t.f}`;
            canvas.style.transformOrigin = "0px 0px";

            // as well as the axis clip box
            // this basically untransforms the clip box by the transform we just applied
            const [left, right] = [(xaxis.scale.range[0] - t.e - viewbox.x) / t.a, (xaxis.scale.range[1] - t.e - viewbox.x) / t.a];
            const [top, bottom] = [(yaxis.scale.range[0] - t.f - viewbox.y) / t.d, (yaxis.scale.range[1] - t.f - viewbox.y) / t.d];
            canvas.style.clipPath = `rect(${top}px ${right}px ${bottom}px ${left}px)`;

            // and we make sure the canvas draws underneath the SVG object
            canvas.style.position = "relative";
            canvas.style.zIndex = "-1";
        }, [xtrans, ytrans]);
    }

    return <g transform={transform.toString()}>
    <foreignObject x={0} y={0} width={width} height={height}>
        <canvas width={width} height={height} ref={canvasRef} style={{imageRendering: "pixelated"}}></canvas>
    </foreignObject>
    </g>;
});

interface PlotLineProps extends React.SVGProps<SVGPathElement> {
    xs: Array<number>
    ys: Array<number>
}

export function PlotLine(props: PlotLineProps) {
    const fig = React.useContext(FigureContext);
    const plot = React.useContext(PlotContext);
    if (fig === undefined || plot === undefined) {
        throw new Error("Component 'PlotLineProps' must be used inside a 'Plot'");
    }

    let xaxis = (typeof plot.xaxis === "string") ? fig.axes.get(plot.xaxis)! : plot.xaxis;
    let yaxis = (typeof plot.yaxis === "string") ? fig.axes.get(plot.yaxis)! : plot.yaxis;

    if (props.xs.length != props.ys.length) {
        throw new Error("In component 'PlotLineProps': `xs` and `ys` must be the same length");
    }

    let path_elems: Array<string> = [];
    let drew_last = false;
    for (let i = 0; i < props.xs.length; i++) {
        const x = xaxis.scale.transform(props.xs[i], false);
        const y = yaxis.scale.transform(props.ys[i], false);
        if (!isFinite(x) || !isFinite(y)) {
            drew_last = false;
            continue
        }
        path_elems.push(
            drew_last ? `L ${x} ${y}` : `M ${x} ${y}`
        );
        drew_last = true;
    }

    const filteredProps: React.SVGProps<SVGPathElement> = {
        fill: "none"
    };

    for (const prop in props) {
        if (["xs", "ys"].includes(prop)) continue;
        filteredProps[prop] = props[prop]
    }

    return <path d={path_elems.join(" ")} className={styles["plot-line"]} {...filteredProps}/>;
}