import { Camera, Sparkles, BookOpen, User, Package } from "lucide-react";
import { useNavigate, useLocation } from "react-router";

export function BottomNav() {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { icon: Camera, label: "掃描", path: "/" },
    { icon: Package, label: "冰箱", path: "/inventory" },
    { icon: Sparkles, label: "AI食譜", path: "/recipes" },
    { icon: BookOpen, label: "紀錄", path: "/saved" },
    { icon: User, label: "我的", path: "/profile" },
  ];

  return (
    <nav className="fixed bottom-0 left-1/2 -translate-x-1/2 z-[100] w-full max-w-[430px] px-4 pb-6 pt-2 pointer-events-none">
      <div className="w-full mx-auto bg-[#0f2e24]/80 backdrop-blur-2xl border border-white/10 rounded-[2.5rem] p-3 flex justify-around items-center shadow-[0_20px_50px_rgba(0,0,0,0.5)] pointer-events-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));

          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`relative flex flex-col items-center gap-1.5 px-5 py-2.5 rounded-2xl transition-all duration-300 group ${isActive ? "text-[#00ff88]" : "text-gray-500 hover:text-white"
                }`}
            >
              {isActive && (
                <div className="absolute inset-0 bg-[#00ff88]/5 rounded-2xl animate-in fade-in zoom-in duration-300" />
              )}
              <Icon size={20} strokeWidth={isActive ? 3 : 2} className="relative z-10" />
              <span className={`text-[8px] font-black tracking-[0.2em] relative z-10 transition-all ${isActive ? "opacity-100 translate-y-0" : "opacity-0 translate-y-1 group-hover:opacity-40"}`}>
                {item.label}
              </span>
              {isActive && (
                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 bg-[#00ff88] rounded-full shadow-[0_0_8px_#00ff88]" />
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
}