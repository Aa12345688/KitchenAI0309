
import { createRoot } from "react-dom/client";
import App from "./app/App.tsx";
import "./styles/index.css"; // 引入全域 Tailwind CSS 樣式

// 獲取 HTML 中的 root 節點並渲染 React 應用程式
createRoot(document.getElementById("root")!).render(<App />);
