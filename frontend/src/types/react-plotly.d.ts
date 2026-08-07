/* eslint-disable @typescript-eslint/no-explicit-any */
declare module 'react-plotly.js' {
  import * as React from 'react';
  import { PlotlyHTMLElement, Layout, Data, Config } from 'plotly.js';

  export interface PlotParams {
    data: Data[];
    layout: Partial<Layout>;
    config?: Partial<Config>;
    frames?: any[];
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: any, graphDiv: PlotlyHTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: PlotlyHTMLElement) => void;
    onPurge?: (figure: any, graphDiv: PlotlyHTMLElement) => void;
    onError?: (err: Error) => void;
  }

  export default class Plot extends React.Component<PlotParams> {}
}
