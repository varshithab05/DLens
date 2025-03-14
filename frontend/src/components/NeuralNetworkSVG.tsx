import React from "react";

const NeuralNetworkSVG: React.FC = () => {
  return (
    <svg width="400" height="400" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg" style={{ background: "transparent" }}>
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={{ stopColor: "#ffffff", stopOpacity: 1 }} />
          <stop offset="100%" style={{ stopColor: "#aaaaaa", stopOpacity: 1 }} />
        </linearGradient>
      </defs>

      <style>
        {`
          .node { fill: url(#grad); stroke: white; stroke-width: 2; }
          .edge {
            stroke: white;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-dasharray: 5, 5;
            animation: move 2s infinite alternate;
          }
          @keyframes move {
            0% { stroke-dashoffset: 10; }
            100% { stroke-dashoffset: 0; }
          }
        `}
      </style>

      {/* Input Layer */}
      {[50, 120, 190, 260, 330].map((y, i) => (
        <circle key={i} className="node" cx="50" cy={y} r="15" />
      ))}

      {/* Hidden Layer */}
      {[75, 150, 225, 300].map((y, i) => (
        <circle key={i} className="node" cx="200" cy={y} r="15" />
      ))}

      {/* Output Layer */}
      {[100, 200, 300].map((y, i) => (
        <circle key={i} className="node" cx="350" cy={y} r="15" />
      ))}

      {/* Connections */}
      {[
        [50, 50, 200, 75],
        [50, 50, 200, 150],
        [50, 120, 200, 75],
        [50, 120, 200, 150],
        [50, 120, 200, 225],
        [50, 190, 200, 150],
        [50, 190, 200, 225],
        [50, 190, 200, 300],
        [50, 260, 200, 225],
        [50, 260, 200, 300],
        [50, 330, 200, 300],
        [200, 75, 350, 100],
        [200, 150, 350, 100],
        [200, 150, 350, 200],
        [200, 225, 350, 200],
        [200, 225, 350, 300],
        [200, 300, 350, 300],
        [350, 100, 400, 100],
        [350, 200, 400, 200],
        [350, 300, 400, 300]
      ].map(([x1, y1, x2, y2], i) => (
        <line key={i} className="edge" x1={x1} y1={y1} x2={x2} y2={y2} />
      ))}
    </svg>
  );
};

export default NeuralNetworkSVG;
