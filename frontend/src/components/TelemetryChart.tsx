'use client';

import React, { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';

// Dynamically import react-plotly.js to avoid SSR errors
const Plot = dynamic(() => import('react-plotly.js'), {
  ssr: false,
  loading: () => (
    <div className="h-[500px] w-full flex flex-col items-center justify-center bg-transparent border border-white/10 rounded-xl text-slate-400">
      <div className="w-12 h-12 border-4 border-slate-700 border-t-emerald-500 rounded-full animate-spin mb-4"></div>
      <p className="font-medium animate-pulse">Initializing Visualization Engine...</p>
    </div>
  )
});

interface TelemetryPoint {
  time: number;
  altitude: number;
  speed: number;
  power_required: number;
  power_delivered: number;
  power_motor: number;
  power_engine: number;
  soc: number;
  fuel: number;
  weight: number;
  phase: string;
  deficit: number;
  u: number;
}

interface TelemetryChartProps {
  telemetry: TelemetryPoint[];
}

export default function TelemetryChart({ telemetry }: TelemetryChartProps) {
  const [activeTab, setActiveTab] = useState<'power' | 'resources' | 'trajectory'>('power');

  // ── Summary stats ────────────────────────────────────────────────────
  const summaryStats = useMemo(() => {
    if (!telemetry.length) return null;
    const maxAlt = Math.max(...telemetry.map(pt => pt.altitude));
    const enduranceH = telemetry[telemetry.length - 1].time / 3600;
    const initialFuel = telemetry[0].fuel;
    const finalFuel = telemetry[telemetry.length - 1].fuel;
    const fuelBurned = initialFuel - finalFuel;
    const maxPower = Math.max(...telemetry.map(pt => pt.power_required));
    return { maxAlt, enduranceH, fuelBurned, maxPower };
  }, [telemetry]);

  if (!telemetry || telemetry.length === 0) {
    return (
      <div className="h-[500px] w-full flex items-center justify-center bg-transparent border border-white/10 rounded-xl text-slate-500 text-sm">
        No simulation data available. Trigger optimization to display charts.
      </div>
    );
  }

  const timeMinutes = telemetry.map((pt) => pt.time / 60);

  const pEngine = telemetry.map((pt) => pt.power_engine);
  const pMotor = telemetry.map((pt) => pt.power_motor);
  const pRequired = telemetry.map((pt) => pt.power_required);
  const pStackedMotor = pEngine.map((engVal, idx) => engVal + pMotor[idx]);

  const socPercent = telemetry.map((pt) => pt.soc * 100);
  const fuelKg = telemetry.map((pt) => pt.fuel);

  const altitudeM = telemetry.map((pt) => pt.altitude);
  const speedMs = telemetry.map((pt) => pt.speed);

  // Phase transition annotations (for power chart)
  const phaseAnnotations: object[] = [];
  const phaseShapes: object[] = [];
  let lastPhase = telemetry[0]?.phase;
  for (let i = 1; i < telemetry.length; i++) {
    if (telemetry[i].phase !== lastPhase) {
      const xVal = timeMinutes[i];
      phaseShapes.push({
        type: 'line',
        x0: xVal, x1: xVal,
        y0: 0, y1: 1,
        yref: 'paper',
        line: { color: 'rgba(148,163,184,0.2)', width: 1, dash: 'dot' },
      });
      phaseAnnotations.push({
        x: xVal,
        y: 1.03,
        yref: 'paper',
        xanchor: 'center',
        text: telemetry[i].phase,
        showarrow: false,
        font: { size: 9, color: '#64748b', family: 'system-ui' },
      });
      lastPhase = telemetry[i].phase;
    }
  }

  const commonLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'system-ui, sans-serif',
      color: '#94a3b8',
    },
    hovermode: 'x unified' as const,
    hoverlabel: {
      bgcolor: '#1e293b',
      bordercolor: '#334155',
      font: { color: '#f8fafc' },
    },
    margin: { t: 48, r: 44, b: 60, l: 64 },
    xaxis: {
      title: { text: 'Time (Minutes)', font: { size: 12 } },
      gridcolor: '#0f172a',
      zerolinecolor: '#1e293b',
      tickfont: { size: 10 },
    },
    yaxis: {
      gridcolor: '#0f172a',
      zerolinecolor: '#1e293b',
      tickfont: { size: 10 },
    },
    showlegend: true,
    legend: {
      orientation: 'h' as const,
      yanchor: 'bottom' as const,
      y: 1.02,
      xanchor: 'right' as const,
      x: 1,
      font: { size: 11 },
    },
    autosize: true,
  };

  const renderChart = () => {
    switch (activeTab) {
      case 'power':
        return (
          <Plot
            data={[
              {
                x: timeMinutes,
                y: pEngine,
                name: 'Turboshaft Power (kW)',
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(59, 130, 246, 0.18)',
                line: { color: '#3b82f6', width: 2 },
                hovertemplate: '%{y:.3f} kW<extra></extra>',
              },
              {
                x: timeMinutes,
                y: pStackedMotor,
                text: pMotor.map(v => v.toFixed(3)),
                name: 'Electric Motor Power (kW)',
                type: 'scatter',
                mode: 'lines',
                fill: 'tonexty',
                fillcolor: 'rgba(255, 180, 84, 0.1)',
                line: { color: '#FFB454', width: 2 },
                hovertemplate: '%{text} kW<extra></extra>',
              },
              {
                x: timeMinutes,
                y: pRequired,
                name: 'Total Power Required (kW)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#f43f5e', width: 2, dash: 'dash' },
                hovertemplate: '%{y:.3f} kW<extra></extra>',
              },
            ]}
            layout={{
              ...commonLayout,
              yaxis: {
                ...commonLayout.yaxis,
                title: { text: 'Power (kW)', font: { size: 12 } },
              },
              shapes: phaseShapes,
              annotations: phaseAnnotations,
            }}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler={true}
            config={{ responsive: true, displayModeBar: false }}
          />
        );

      case 'resources':
        return (
          <Plot
            data={[
              {
                x: timeMinutes,
                y: socPercent,
                name: 'Battery State of Charge (%)',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#FFB454', width: 2.5 },
              },
              {
                x: timeMinutes,
                y: fuelKg,
                name: 'Fuel Remaining (kg)',
                type: 'scatter',
                mode: 'lines',
                yaxis: 'y2',
                line: { color: '#f97316', width: 2.5 },
              },
            ]}
            layout={{
              ...commonLayout,
              yaxis: {
                ...commonLayout.yaxis,
                title: { text: 'Battery SoC (%)', font: { size: 12, color: '#FFB454' } },
                tickfont: { color: '#FFB454' },
              },
              yaxis2: {
                title: { text: 'Fuel Remaining (kg)', font: { size: 12, color: '#f97316' } },
                tickfont: { color: '#f97316' },
                overlaying: 'y',
                side: 'right',
                gridcolor: 'transparent',
              },
            }}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler={true}
            config={{ responsive: true, displayModeBar: false }}
          />
        );

      case 'trajectory':
        return (
          <Plot
            data={[
              {
                x: timeMinutes,
                y: altitudeM,
                name: 'Altitude (m)',
                type: 'scatter',
                mode: 'lines',
                fill: 'tozeroy',
                fillcolor: 'rgba(99, 102, 241, 0.08)',
                line: { color: '#6366f1', width: 2.5 },
              },
              {
                x: timeMinutes,
                y: speedMs,
                name: 'Speed (m/s)',
                type: 'scatter',
                mode: 'lines',
                yaxis: 'y2',
                line: { color: '#a855f7', width: 2, dash: 'dot' },
              },
            ]}
            layout={{
              ...commonLayout,
              yaxis: {
                ...commonLayout.yaxis,
                title: { text: 'Altitude (m)', font: { size: 12, color: '#6366f1' } },
                tickfont: { color: '#6366f1' },
              },
              yaxis2: {
                title: { text: 'Speed (m/s)', font: { size: 12, color: '#a855f7' } },
                tickfont: { color: '#a855f7' },
                overlaying: 'y',
                side: 'right',
                gridcolor: 'transparent',
              },
            }}
            style={{ width: '100%', height: '500px' }}
            useResizeHandler={true}
            config={{ responsive: true, displayModeBar: false }}
          />
        );
    }
  };

  const tabs: { key: 'power' | 'resources' | 'trajectory'; label: string }[] = [
    { key: 'power', label: 'Power Split' },
    { key: 'resources', label: 'Resource Status' },
    { key: 'trajectory', label: 'Flight Profile' },
  ];

  return (
    <div
      className="w-full rounded-xl border border-white/10/60 overflow-hidden"
      style={{
        background: 'transparent',
        borderTopWidth: '2px',
        borderTopColor: '#1F2733',
      }}
    >
      {/* Title row */}
      <div className="px-6 pt-5 pb-2">
        <h2 className="text-base font-semibold text-slate-100">Mission Telemetry Visualizer</h2>
        <p className="text-xs text-slate-400 mt-0.5">Interactive telemetry timeline generated by physics engine</p>
      </div>

      {/* ── Summary stats bar ──────────────────────────────────────────── */}
      {summaryStats && (
        <div className="mx-6 mb-4 grid grid-cols-4 gap-2">
          {[
            { label: 'Max Altitude', value: `${summaryStats.maxAlt.toFixed(0)} m`, color: 'text-indigo-400' },
            { label: 'Endurance', value: `${summaryStats.enduranceH.toFixed(2)} h`, color: 'text-[#FFB454]' },
            { label: 'Fuel Burned', value: `${summaryStats.fuelBurned.toFixed(1)} kg`, color: 'text-orange-400' },
            { label: 'Peak Power', value: `${summaryStats.maxPower.toFixed(1)} kW`, color: 'text-red-400' },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              className="bg-white/5 backdrop-blur-md border border-white/10/50 rounded-lg px-3 py-2"
            >
              <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">{label}</p>
              <p className={`text-sm font-bold font-mono ${color}`}>{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Tab buttons ──────────────────────────────────────────────── */}
      <div className="px-6 flex items-center gap-1 border-b border-white/10/60 mb-0">
        <div className="flex border border-white/10/60 rounded-lg p-0.5 bg-white/5 backdrop-blur-md">
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`relative px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                activeTab === key
                  ? 'text-[#FFB454] bg-slate-800 shadow-none'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
              style={activeTab === key ? { textShadow: 'none' } : {}}
            >
              {label}
              {activeTab === key && (
                <span className="absolute bottom-0 left-2 right-2 h-0.5 bg-[#FFB454] rounded-full" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="w-full px-0">{renderChart()}</div>
    </div>
  );
}
