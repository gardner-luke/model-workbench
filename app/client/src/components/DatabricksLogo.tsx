interface DatabricksLogoProps {
  className?: string;
  size?: number;
}

// Stylized Databricks "lava" mark — five stacked rectangles in lava red.
export function DatabricksLogo({ className = '', size = 28 }: DatabricksLogoProps) {
  return (
    <svg
      viewBox="0 0 32 32"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label="Databricks"
    >
      <defs>
        <linearGradient id="dbx-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FF3621" />
          <stop offset="100%" stopColor="#E62D1A" />
        </linearGradient>
      </defs>
      <g fill="url(#dbx-grad)">
        <rect x="2" y="4" width="28" height="2.5" rx="0.5" />
        <rect x="3" y="9" width="26" height="2.5" rx="0.5" />
        <rect x="4" y="14" width="24" height="2.5" rx="0.5" />
        <rect x="5" y="19" width="22" height="2.5" rx="0.5" />
        <rect x="6" y="24" width="20" height="2.5" rx="0.5" />
      </g>
    </svg>
  );
}
