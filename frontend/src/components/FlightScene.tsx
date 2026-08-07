'use client';

import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, OrthographicCamera, Grid, Html, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */
interface TelemetryPoint {
  time: number;
  altitude: number;
  speed: number;
  phase: string;
  u: number;
  climb_rate: number;
}

interface FlightSceneProps {
  telemetry: TelemetryPoint[];
  currentIndex: number;
}

/* ------------------------------------------------------------------ */
/*  Generate 3D Flight Path from Telemetry                             */
/*  X = cumulative horizontal distance (scaled)                        */
/*  Y = altitude (scaled)                                              */
/*  Z = lateral offset (racetrack during loiter)                       */
/* ------------------------------------------------------------------ */
function generateFlightPath(telemetry: TelemetryPoint[]): THREE.Vector3[] {
  if (telemetry.length === 0) return [];

  const points: THREE.Vector3[] = [];
  let cumDist = 0;
  // ALT_SCALE reduced from 0.015 → 0.003 so 5000m = 15 scene units.
  // DIST_SCALE increased from 0.0008 → 0.001.
  // Old ratio (ALT/DIST) = 18.75 made climb look 70° vertical.
  // New ratio = 3.0 → apparent climb angle ≈ 23°, which looks realistic.
  const ALT_SCALE = 0.003;
  const DIST_SCALE = 0.001;

  // Pre-scan: find loiter window
  let loiterStartIdx = -1;
  let loiterEndIdx = -1;
  for (let i = 0; i < telemetry.length; i++) {
    if (telemetry[i].phase === 'loiter') {
      if (loiterStartIdx < 0) loiterStartIdx = i;
      loiterEndIdx = i;
    }
  }

  // Pre-scan: find descent + landing window for return-path lerp
  let descentStartIdx = -1;
  let landingEndIdx = -1;
  for (let i = 0; i < telemetry.length; i++) {
    if ((telemetry[i].phase === 'descent' || telemetry[i].phase === 'landing')) {
      if (descentStartIdx < 0) descentStartIdx = i;
      landingEndIdx = i;
    }
  }
  const returnDuration = descentStartIdx >= 0 && landingEndIdx >= 0
    ? Math.max(1, telemetry[landingEndIdx].time - telemetry[descentStartIdx].time)
    : 1;

  // Compute loiter center X by accumulating distance to loiter start
  let loiterCenterX = 0;
  let loiterStartTime = 0;
  let loiterDuration = 1;
  if (loiterStartIdx >= 0) {
    let d = 0;
    for (let j = 1; j <= loiterStartIdx; j++) {
      const dt = telemetry[j].time - telemetry[j - 1].time;
      const spd = telemetry[j].speed;
      const cr = telemetry[j].climb_rate || 0;
      d += spd * Math.cos(Math.asin(Math.max(-1, Math.min(1, cr / Math.max(spd, 1))))) * dt;
    }
    loiterCenterX = d * DIST_SCALE;
    loiterStartTime = telemetry[loiterStartIdx].time;
    loiterDuration = Math.max(1, telemetry[loiterEndIdx].time - loiterStartTime);
  }

  // Racetrack parameters: exactly 3 orbits, drifting forward
  // Radii scaled down from 12/20 → 6/10 to match the reduced ALT_SCALE
  const NUM_ORBITS = 3;
  const ORBIT_RADIUS_Z = 6;
  const ORBIT_RADIUS_X = 10;
  const FORWARD_DRIFT = 8; // total forward distance across all orbits

  // Return lane parallel to outbound path — same Z offset as orbit radius
  const RETURN_LANE_Z = ORBIT_RADIUS_Z; // = 6 scene units

  // Descent/landing return path state (captured from last loiter point)
  let returnStartX = 0;
  let returnStartZ = 0;
  let returnStartTime = 0;

  for (let i = 0; i < telemetry.length; i++) {
    const pt = telemetry[i];
    const dt = i > 0 ? pt.time - telemetry[i - 1].time : 0;
    const horizSpeed = pt.speed * Math.cos(Math.asin(Math.max(-1, Math.min(1, (pt.climb_rate || 0) / Math.max(pt.speed, 1)))));
    cumDist += horizSpeed * dt;

    const y = pt.altitude * ALT_SCALE;

    if (pt.phase === 'loiter' && loiterStartIdx >= 0) {
      // Forward-drifting racetrack: exactly NUM_ORBITS orbits
      const loiterElapsed = pt.time - loiterStartTime;
      const t = loiterElapsed / loiterDuration; // 0..1 over loiter phase
      const theta = 2 * Math.PI * NUM_ORBITS * t;
      const drift = FORWARD_DRIFT * t; // gradual forward progress
      const x = loiterCenterX + drift + Math.cos(theta) * ORBIT_RADIUS_X;
      const z = Math.sin(theta) * ORBIT_RADIUS_Z;
      points.push(new THREE.Vector3(x, y, z));

    } else if ((pt.phase === 'descent' || pt.phase === 'landing') && loiterStartIdx >= 0) {
      // Return to origin along a parallel lane at Z = +RETURN_LANE_Z
      if (pt.phase === 'descent' && (i === 0 || telemetry[i - 1].phase !== 'descent')) {
        // Capture exit state from the last loiter point
        const lastPt = points[points.length - 1];
        returnStartX = lastPt ? lastPt.x : 0;
        returnStartZ = lastPt ? lastPt.z : 0;
        returnStartTime = pt.time;
      }

      // Progress 0→1 over the entire descent+landing duration
      const returnElapsed = pt.time - returnStartTime;
      const tReturn = Math.min(1, returnElapsed / returnDuration);

      // Cosine ease-in-out for smooth deceleration at both ends
      const ease = (1 - Math.cos(tReturn * Math.PI)) / 2;

      // X: lerp from loiter exit X back to 0 (origin)
      const x = returnStartX * (1 - ease);

      // Z: first snap to return lane (RETURN_LANE_Z), then hold it back to origin
      // Phase 1 (0→0.15): sweep Z from loiter exit Z to RETURN_LANE_Z
      // Phase 2 (0.15→1): hold Z = RETURN_LANE_Z while heading home
      const zEase = Math.min(1, tReturn / 0.15);
      const zSweep = (1 - Math.cos(zEase * Math.PI)) / 2;
      const z = returnStartZ + (RETURN_LANE_Z - returnStartZ) * zSweep;

      points.push(new THREE.Vector3(x, y, z));

    } else {
      // Straight line: takeoff, climb, cruise
      const x = cumDist * DIST_SCALE;
      points.push(new THREE.Vector3(x, y, 0));
    }
  }

  return points;
}

/* ------------------------------------------------------------------ */
/*  Color-code segments by PSR (power source indicator)                */
/* ------------------------------------------------------------------ */
function generatePathColors(telemetry: TelemetryPoint[]): THREE.Color[] {
  const COLOR_BATTERY = new THREE.Color(0xF59E0B);
  const COLOR_HYBRID  = new THREE.Color(0x06B6D4);
  const COLOR_ENGINE  = new THREE.Color(0xFFB454);
  const COLOR_IDLE    = new THREE.Color(0x4B5563);

  return telemetry.map((pt) => {
    if (pt.phase === 'descent' || pt.phase === 'landing') return COLOR_IDLE;
    if (pt.u > 0.4) return COLOR_BATTERY;
    if (pt.u > 0.15) return COLOR_HYBRID;
    return COLOR_ENGINE;
  });
}

/* ------------------------------------------------------------------ */
/*  Phase transition markers (waypoints)                               */
/* ------------------------------------------------------------------ */
function PhaseMarkers({ telemetry, flightPath }: { telemetry: TelemetryPoint[]; flightPath: THREE.Vector3[] }) {
  const markers = useMemo(() => {
    if (telemetry.length === 0 || flightPath.length === 0) return [];
    const result: { phase: string; position: THREE.Vector3; altitude: number }[] = [];
    let prevPhase = '';
    for (let i = 0; i < telemetry.length; i++) {
      if (telemetry[i].phase !== prevPhase && telemetry[i].phase !== 'completed') {
        result.push({
          phase: telemetry[i].phase,
          position: flightPath[i] || new THREE.Vector3(),
          altitude: telemetry[i].altitude,
        });
        prevPhase = telemetry[i].phase;
      }
    }
    return result;
  }, [telemetry, flightPath]);

  const phaseColors: Record<string, string> = {
    takeoff: '#EF4444', climb: '#F59E0B', cruise: '#06B6D4',
    loiter: '#8B5CF6', descent: '#14B8A6', landing: '#FFB454',
  };

  return (
    <>
      {markers.map((m, i) => (
        <group key={i} position={m.position}>
          {/* Small sphere at waypoint */}
          <mesh>
            <sphereGeometry args={[0.5, 8, 8]} />
            <meshStandardMaterial
              color={phaseColors[m.phase] || '#9CA3AF'}
              emissive={phaseColors[m.phase] || '#9CA3AF'}
              emissiveIntensity={0.5}
            />
          </mesh>
          {/* Phase label */}
          <Html
            position={[0, 2.0, 0]}
            center
            style={{ pointerEvents: 'none' }}
          >
            <div style={{
              background: '#12161FE0',
              border: `1px solid ${phaseColors[m.phase] || '#374151'}`,
              borderRadius: '3px',
              padding: '2px 6px',
              whiteSpace: 'nowrap',
              fontSize: '9px',
              fontFamily: 'monospace',
              fontWeight: 700,
              color: phaseColors[m.phase] || '#9CA3AF',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              {m.phase} · {m.altitude.toFixed(0)}m
            </div>
          </Html>
        </group>
      ))}
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  UAV Marker with HUD label                                          */
/* ------------------------------------------------------------------ */
function UAVMarker({ position, nextPosition, telemetryPt }: {
  position: THREE.Vector3;
  nextPosition: THREE.Vector3 | null;
  telemetryPt: TelemetryPoint | null;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current) return;
    meshRef.current.position.copy(position);
    if (nextPosition) {
      const dir = new THREE.Vector3().subVectors(nextPosition, position).normalize();
      if (dir.length() > 0.001) {
        const q = new THREE.Quaternion();
        q.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        meshRef.current.quaternion.copy(q);
      }
    }
  });

  return (
    <group>
      <mesh ref={meshRef} position={position}>
        <coneGeometry args={[1.8, 5, 4]} />
        <meshStandardMaterial color="#F59E0B" emissive="#F59E0B" emissiveIntensity={0.6} flatShading />
      </mesh>

      {/* HUD Label floating above UAV */}
      {telemetryPt && (
        <Html
          position={[position.x, position.y + 3, position.z]}
          center
          style={{ pointerEvents: 'none' }}
        >
          <div style={{
            background: '#0A0E14D0',
            border: '1px solid #F59E0B50',
            borderRadius: '4px',
            padding: '4px 8px',
            whiteSpace: 'nowrap',
            fontSize: '10px',
            fontFamily: 'monospace',
            color: '#E5E7EB',
            display: 'flex',
            flexDirection: 'column',
            gap: '1px',
            minWidth: '120px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
              <span style={{ color: '#9CA3AF', fontSize: '8px' }}>ALT</span>
              <span style={{ color: '#FFB454', fontWeight: 700 }}>{telemetryPt.altitude.toFixed(0)}m</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
              <span style={{ color: '#9CA3AF', fontSize: '8px' }}>TAS</span>
              <span style={{ color: '#06B6D4', fontWeight: 700 }}>{(telemetryPt.speed * 3.6).toFixed(0)} km/h</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
              <span style={{ color: '#9CA3AF', fontSize: '8px' }}>PSR</span>
              <span style={{ color: telemetryPt.u > 0.3 ? '#F59E0B' : '#FFB454', fontWeight: 700 }}>
                {(telemetryPt.u * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

/* ------------------------------------------------------------------ */
/*  Altitude Reference Line (dashed drop to ground)                    */
/* ------------------------------------------------------------------ */
function AltitudeIndicator({ position }: { position: THREE.Vector3 }) {
  const { x, y, z } = position;
  const points = useMemo(() => {
    return [new THREE.Vector3(x, y, z), new THREE.Vector3(x, 0, z)];
  }, [x, y, z]);

  return (
    <Line points={points} color="#F59E0B" lineWidth={1} dashed dashSize={0.8} gapSize={0.5} />
  );
}

/* ------------------------------------------------------------------ */
/*  Axis Labels (distance & altitude)                                  */
/* ------------------------------------------------------------------ */
function AxisLabels() {
  return (
    <group>
      <mesh position={[20, 0, 0]}>
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshStandardMaterial color="#EF4444" emissive="#EF4444" emissiveIntensity={0.4} />
      </mesh>
      <Html position={[24, 0, 0]} center style={{ pointerEvents: 'none' }}>
        <span style={{ fontSize: '9px', fontFamily: 'monospace', color: '#EF4444', fontWeight: 700 }}>
          DIST →
        </span>
      </Html>

      <mesh position={[0, 20, 0]}>
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshStandardMaterial color="#FFB454" emissive="#FFB454" emissiveIntensity={0.4} />
      </mesh>
      <Html position={[0, 23, 0]} center style={{ pointerEvents: 'none' }}>
        <span style={{ fontSize: '9px', fontFamily: 'monospace', color: '#FFB454', fontWeight: 700 }}>
          ALT ↑
        </span>
      </Html>
    </group>
  );
}

/* ------------------------------------------------------------------ */
/*  Terrain Wireframe Map                                              */
/* ------------------------------------------------------------------ */
function TerrainGrid() {
  const geom = useMemo(() => {
    const width = 6000;
    const height = 240;
    const wSegments = 120;
    const hSegments = 20;
    // Align plane horizontally (along X and Z)
    const g = new THREE.PlaneGeometry(width, height, wSegments, hSegments);
    
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      
      // Multi-frequency sine-cosine waves for natural looking low-poly tactical hills
      const zVal = Math.sin(x * 0.01) * Math.cos(y * 0.02) * 6 +
                   Math.sin(x * 0.003) * 12 +
                   Math.cos(y * 0.04) * 3;
      pos.setZ(i, zVal);
    }
    g.computeVertexNormals();
    return g;
  }, []);

  return (
    <mesh geometry={geom} rotation={[-Math.PI / 2, 0, 0]} position={[2500, -6, 0]}>
      <meshBasicMaterial color="#1E293B" wireframe transparent opacity={0.16} />
    </mesh>
  );
}

/* ------------------------------------------------------------------ */
/*  Tactical Space Dome & Static Starfield Background                  */
/* ------------------------------------------------------------------ */
function TacticalBackground({ currentX }: { currentX: number }) {
  const starPositions = useMemo(() => {
    const pts: number[] = [];
    const count = 500;
    // Simple pure seeded LCG generator to prevent impure React 19 render warnings
    // and Next.js hydration mismatches.
    let seed = 42;
    const lcg = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };

    // Generate static stars scattered along the entire flight corridor once
    for (let i = 0; i < count; i++) {
      const x = -500 + lcg() * 8000; // wider corridor
      const y = 18 + lcg() * 80;  // lowered: was 30+220, cruise is now at y=15
      const z = -150 + lcg() * 300; // lateral scatter
      pts.push(x, y, z);
    }
    return new Float32Array(pts);
  }, []);

  return (
    <group>
      {/* High-density radar tracking background dots */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[starPositions, 3]}
            count={starPositions.length / 3}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          color="#4B5563"
          size={1.3}
          sizeAttenuation={true}
          transparent
          opacity={0.5}
        />
      </points>

      {/* Subtle coordinate dome overlay following the UAV */}
      <mesh position={[currentX, 0, 0]}>
        <sphereGeometry args={[250, 20, 14, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshBasicMaterial color="#1E293B" wireframe transparent opacity={0.03} />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------ */
/*  Main 3D Scene (inner Canvas content)                               */
/* ------------------------------------------------------------------ */
function SceneContent({ telemetry, currentIndex }: FlightSceneProps) {
  const flightPath = useMemo(() => generateFlightPath(telemetry), [telemetry]);
  const pathColors = useMemo(() => generatePathColors(telemetry), [telemetry]);

  const currentPos = useMemo(() => {
    if (flightPath.length === 0) return new THREE.Vector3(0, 0, 0);
    const idx = Math.min(currentIndex, flightPath.length - 1);
    return flightPath[idx];
  }, [flightPath, currentIndex]);

  const nextPos = useMemo(() => {
    if (flightPath.length === 0) return null;
    const idx = Math.min(currentIndex + 1, flightPath.length - 1);
    return flightPath[idx];
  }, [flightPath, currentIndex]);

  const currentTelemetry = useMemo(() => {
    if (telemetry.length === 0) return null;
    return telemetry[Math.min(currentIndex, telemetry.length - 1)];
  }, [telemetry, currentIndex]);

  // Trail and future paths
  const trailPath = useMemo(() => {
    return flightPath.slice(0, Math.min(currentIndex + 1, flightPath.length));
  }, [flightPath, currentIndex]);

  const trailColors = useMemo(() => {
    return pathColors.slice(0, Math.min(currentIndex + 1, pathColors.length));
  }, [pathColors, currentIndex]);

  const futurePath = useMemo(() => {
    return flightPath.slice(Math.min(currentIndex, flightPath.length - 1));
  }, [flightPath, currentIndex]);

  if (flightPath.length === 0) return null;

  return (
    <>
      {/* Camera moved to [120, 45, 130]: lower elevation gives side-profile view.
           Zoom increased 3→6 to compensate for the smaller vertical scale (15 vs 75 units). */}
      <OrthographicCamera makeDefault position={[120, 45, 130]} zoom={6.0} near={0.1} far={8000} />

      {/* Orbit controls — gentle tilt/pan, open vertical rotation range */}
      <OrbitControls
        target={[currentPos.x, currentPos.y * 0.4, currentPos.z]}
        enableDamping
        dampingFactor={0.08}
        enablePan
        panSpeed={0.6}
        enableZoom
        zoomSpeed={0.8}
        minZoom={0.8}
        maxZoom={20}
        minPolarAngle={Math.PI / 18}   /* 10° — allows viewing from almost side-on profile */
        maxPolarAngle={Math.PI / 2.15} /* 83° — prevents rotating below the ground plane */
        enableRotate
        rotateSpeed={0.45}
      />

      {/* Lighting */}
      <ambientLight intensity={0.55} />
      <hemisphereLight color="#1E293B" groundColor="#0F172A" intensity={0.7} />
      <directionalLight position={[50, 100, 50]} intensity={1.5} color="#F3F4F6" />

      {/* Stars and space dome background centered around current position */}
      <TacticalBackground currentX={currentPos.x} />

      {/* Wireframe Terrain Grid representing the landscape below the flight corridor */}
      <TerrainGrid />

      {/* Ground Grid */}
      <Grid
        args={[500, 500]}
        cellSize={5}
        cellThickness={0.3}
        cellColor="#1F2937"
        sectionSize={25}
        sectionThickness={0.6}
        sectionColor="#374151"
        fadeDistance={350}
        position={[currentPos.x, 0, 0]}
      />

      <AxisLabels />

      {/* Future path — dim */}
      {futurePath.length >= 2 && (
        <Line points={futurePath} color="#374151" lineWidth={1.5} />
      )}

      {/* Flown trail — color-coded by PSR */}
      {trailPath.length >= 2 && (
        <Line
          points={trailPath}
          vertexColors={trailColors.map(c => [c.r, c.g, c.b] as [number, number, number])}
          lineWidth={2.5}
        />
      )}

      {/* Phase transition waypoint markers */}
      <PhaseMarkers telemetry={telemetry} flightPath={flightPath} />

      {/* Altitude indicator dashed line */}
      <AltitudeIndicator position={currentPos} />

      {/* UAV marker with live HUD */}
      <UAVMarker position={currentPos} nextPosition={nextPos} telemetryPt={currentTelemetry} />
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  Exported Component                                                 */
/* ------------------------------------------------------------------ */
export default function FlightScene({ telemetry, currentIndex }: FlightSceneProps) {
  if (!telemetry || telemetry.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-[#0A0E14] text-slate-500 text-xs font-mono">
        AWAITING TELEMETRY DATA...
      </div>
    );
  }

  const currentPt = telemetry[Math.min(currentIndex, telemetry.length - 1)];
  const alt = currentPt?.altitude || 0;
  
  // Generate tape tick marks
  const tickSpacing = 40; // pixels per 100m
  const centerOffset = (alt % 100) / 100 * tickSpacing;
  const baseAlt = Math.floor(alt / 100) * 100;
  
  const ticks = [];
  for(let i = 3; i >= -3; i--) {
    ticks.push(baseAlt + i * 100);
  }

  return (
    <div className="relative w-full h-full">
      <Canvas
        gl={{ antialias: true, alpha: false }}
        style={{ background: 'transparent', position: 'absolute', inset: 0 }}
        dpr={[1, 1.5]}
      >
        <SceneContent telemetry={telemetry} currentIndex={currentIndex} />
      </Canvas>
      
      {/* Signature Element: MIL-STD Altitude Tape */}
      <div className="absolute right-6 top-1/2 -translate-y-1/2 w-16 h-64 border-l-2 border-white/10 bg-black/20 backdrop-blur-md pointer-events-none overflow-hidden font-mono text-[10px] text-[#E8EDF2] flex flex-col items-start select-none shadow-[-8px_0_16px_rgba(10,14,20,0.5)]">
        {/* The Tape Container */}
        <div className="absolute w-full h-full" style={{ transform: `translateY(${centerOffset}px)` }}>
          {ticks.map((val, idx) => (
            <div key={idx} className="absolute w-full flex items-center gap-1" style={{ top: `calc(50% - ${(3 - idx) * tickSpacing}px)` }}>
              <div className="w-2 h-px bg-[#5C6773]" />
              <span className={val === Math.round(alt/100)*100 && (alt%100 < 5 || alt%100 > 95) ? 'text-[#FFB454] font-bold' : 'text-[#5C6773]'}>
                {val}
              </span>
            </div>
          ))}
        </div>
        {/* Center reticle */}
        <div className="absolute top-1/2 -translate-y-1/2 right-0 left-0 h-7 border border-[#FFB454] bg-white/10 flex items-center justify-between px-1">
          <div className="w-1.5 h-1.5 bg-[#FFB454]" />
          <span className="font-bold text-[#FFB454] text-[11px]">{alt.toFixed(0)}</span>
        </div>
        {/* Title */}
        <div className="absolute top-1 right-1 text-[8px] text-[#5C6773] font-bold bg-white/5 px-1 rounded-sm border border-white/10">ALT M</div>
      </div>
    </div>
  );
}

