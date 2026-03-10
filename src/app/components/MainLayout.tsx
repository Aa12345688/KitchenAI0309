import { Outlet, useLocation, useNavigate } from "react-router";
import { BottomNav } from "../components/BottomNav";
import { motion, AnimatePresence } from "framer-motion";
import { useIngredients } from "../services/IngredientContext";

export function MainLayout() {
    const location = useLocation();
    const navigate = useNavigate();
    const { settings } = useIngredients();

    const tabs = ["/", "/inventory", "/recipes", "/saved", "/profile"];
    const currentIndex = tabs.findIndex(t => t === location.pathname || (t !== '/' && location.pathname.startsWith(t)));

    const handleDragEnd = (_: any, info: { offset: { x: number } }) => {
        const threshold = 100;
        if (info.offset.x > threshold && currentIndex > 0) {
            navigate(tabs[currentIndex - 1]);
        } else if (info.offset.x < -threshold && currentIndex < tabs.length - 1 && currentIndex !== -1) {
            navigate(tabs[currentIndex + 1]);
        }
    };

    return (
        <div className={`min-h-screen bg-black flex justify-center w-full ${!settings.darkMode ? 'light-theme' : ''}`}>
            <div className="w-full max-w-[430px] min-h-screen bg-[#0f2e24] text-white relative flex flex-col shadow-2xl overflow-hidden filter-theme">
                <main className="flex-1 overflow-y-auto no-scrollbar scroll-smooth">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={location.pathname}
                            drag="x"
                            dragConstraints={{ left: 0, right: 0 }}
                            onDragEnd={handleDragEnd}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.3, ease: "easeOut" }}
                            className="touch-pan-y min-h-full"
                        >
                            <Outlet />
                        </motion.div>
                    </AnimatePresence>
                </main>
                <BottomNav />
            </div>
        </div>
    );
}
