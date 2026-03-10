import { useNavigate } from "react-router";
import { ArrowLeft, Menu, Sparkles } from "lucide-react";
import { ReactNode } from "react";

// --- PageHeader ---
interface PageHeaderProps {
    showBackButton?: boolean;
    title?: string;
    rightAction?: ReactNode;
}

export function PageHeader({ showBackButton = false, title = "KITCHEN AI", rightAction }: PageHeaderProps) {
    const navigate = useNavigate();

    return (
        <header className="flex items-center justify-between px-4 py-4 sticky top-0 bg-[#0f2e24]/80 backdrop-blur-md z-50">
            <div className="flex items-center gap-2">
                {showBackButton ? (
                    <button onClick={() => navigate(-1)} className="p-2 -ml-2 hover:bg-white/5 rounded-full transition-colors">
                        <ArrowLeft size={24} />
                    </button>
                ) : (
                    <button className="p-2 -ml-2 hover:bg-white/5 rounded-full transition-colors">
                        <Menu size={24} />
                    </button>
                )}
            </div>

            <h1 className="text-lg flex items-center gap-2 font-bold tracking-tight">
                <Sparkles size={20} className="text-[#00ff88]" />
                {title.toUpperCase()}
            </h1>

            <div className="flex items-center gap-2">
                {rightAction || (
                    <button className="p-2 -mr-2 hover:bg-white/5 rounded-full transition-colors">
                        <div className="w-8 h-8 rounded-full bg-gray-600 border border-gray-500 shadow-inner" />
                    </button>
                )}
            </div>
        </header>
    );
}

// --- ImageWithFallback ---
// (We will add more shared components here as we find them)
