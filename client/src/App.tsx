import { useEffect, useState } from 'react';
import { createBrowserRouter, RouterProvider, NavLink, Outlet } from 'react-router';
import { BarChart3 } from 'lucide-react';
import { RegistryPage } from './pages/Registry';
import { PlaygroundPage } from './pages/Playground';
import { EmbeddingsPage } from './pages/Embeddings';
import { SegmentationPage } from './pages/Segmentation';
import { DetectionPage } from './pages/Detection';
import { DepthPage } from './pages/Depth';
import { DatabricksLogo } from './components/DatabricksLogo';

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
    isActive
      ? 'bg-foreground text-background'
      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
  }`;

function Layout() {
  const [dashboardUrl, setDashboardUrl] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    fetch('/api/workspace')
      .then((r) => r.json() as Promise<{ dashboardUrl?: string }>)
      .then((data) => {
        if (!cancelled) setDashboardUrl(data.dashboardUrl ?? '');
      })
      .catch(() => {
        // non-blocking — analytics link is hidden if endpoint fails
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <header className="border-b bg-white">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-6">
          <div className="flex items-center gap-2.5">
            <DatabricksLogo size={26} />
            <div className="flex items-baseline gap-2">
              <span className="text-base font-semibold text-foreground tracking-tight">
                Model Workbench
              </span>
              <span className="text-xs text-muted-foreground">on Databricks</span>
            </div>
          </div>
          <nav className="flex gap-1 items-center">
            <NavLink to="/" end className={navLinkClass}>
              Registry
            </NavLink>
            {dashboardUrl && (
              <a
                href={dashboardUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1.5 rounded-md text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors inline-flex items-center gap-1.5"
              >
                <BarChart3 className="h-4 w-4" /> Analytics
              </a>
            )}
          </nav>
        </div>
      </header>
      <main className="flex-1 px-6 py-8">
        <Outlet />
      </main>
      <footer className="border-t bg-muted/30">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between text-xs text-muted-foreground">
          <span>Built on Databricks AppKit · Foundation Model APIs · Custom GPU Model Serving</span>
          <span className="font-mono">model-workbench</span>
        </div>
      </footer>
    </div>
  );
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <RegistryPage /> },
      { path: '/playground/:name', element: <PlaygroundPage /> },
      { path: '/embeddings/:name', element: <EmbeddingsPage /> },
      { path: '/segmentation/:name', element: <SegmentationPage /> },
      { path: '/detection/:name', element: <DetectionPage /> },
      { path: '/depth/:name', element: <DepthPage /> },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
