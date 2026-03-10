import { createBrowserRouter } from "react-router";
import { lazy, Suspense } from "react";
import { MainLayout } from "./components/MainLayout";

// Lazy load all page components for code splitting
// Lazy load components from the consolidated Views file
const Scanner = lazy(() => import("./pages/Views").then(m => ({ default: m.Scanner })));
const Ingredients = lazy(() => import("./pages/Views").then(m => ({ default: m.Ingredients })));
const Recipes = lazy(() => import("./pages/Views").then(m => ({ default: m.Recipes })));
const RecipeDetail = lazy(() => import("./pages/Views").then(m => ({ default: m.RecipeDetail })));
const Saved = lazy(() => import("./pages/Views").then(m => ({ default: m.Saved })));
const Profile = lazy(() => import("./pages/Views").then(m => ({ default: m.Profile })));
const Inventory = lazy(() => import("./pages/Views").then(m => ({ default: m.Inventory })));

// Loading component for suspense
const PageLoader = () => (
  <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6">
    <div className="relative w-16 h-16">
      <div className="absolute inset-0 border-2 border-[#00ff88]/20 rounded-full animate-[ping_2s_linear_infinite]" />
      <div className="absolute inset-0 border-2 border-[#00ff88]/40 rounded-full animate-[ping_1.5s_linear_infinite]" />
      <div className="absolute inset-0 border-r-2 border-[#00ff88] rounded-full animate-spin shadow-[0_0_15px_#00ff88]" />
      <div className="absolute inset-4 bg-[#00ff88]/10 rounded-full animate-pulse blur-sm" />
    </div>
    <div className="flex flex-col items-center gap-1">
      <span className="text-[10px] font-black text-[#00ff88] tracking-[0.4em] uppercase animate-pulse">
        Synchronizing
      </span>
      <div className="flex gap-1">
        <div className="w-1 h-1 bg-[#00ff88] rounded-full animate-bounce [animation-delay:-0.3s]" />
        <div className="w-1 h-1 bg-[#00ff88] rounded-full animate-bounce [animation-delay:-0.15s]" />
        <div className="w-1 h-1 bg-[#00ff88] rounded-full animate-bounce" />
      </div>
    </div>
  </div>
);

// 定義應用程式的所有路徑與對應的頁面組件
export const router = createBrowserRouter([
  {
    path: "/",
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: (
          <Suspense fallback={<PageLoader />}>
            <Scanner />
          </Suspense>
        ),
      },
      {
        path: "ingredients",
        element: (
          <Suspense fallback={<PageLoader />}>
            <Ingredients />
          </Suspense>
        ),
      },
      {
        path: "recipes",
        element: (
          <Suspense fallback={<PageLoader />}>
            <Recipes />
          </Suspense>
        ),
      },
      {
        path: "recipe/:id",
        element: (
          <Suspense fallback={<PageLoader />}>
            <RecipeDetail />
          </Suspense>
        ),
      },
      {
        path: "saved",
        element: (
          <Suspense fallback={<PageLoader />}>
            <Saved />
          </Suspense>
        ),
      },
      {
        path: "profile",
        element: (
          <Suspense fallback={<PageLoader />}>
            <Profile />
          </Suspense>
        ),
      },
      {
        path: "inventory",
        element: (
          <Suspense fallback={<PageLoader />}>
            <Inventory />
          </Suspense>
        ),
      },
    ],
  },
]);