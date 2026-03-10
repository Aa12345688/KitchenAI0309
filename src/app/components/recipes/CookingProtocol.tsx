interface Step {
    title: string;
    description: string;
}

interface CookingProtocolProps {
    steps: Step[];
}

export function CookingProtocol({ steps }: CookingProtocolProps) {
    return (
        <div className="mb-10 px-2">
            <h3 className="text-[10px] font-black text-gray-400 uppercase tracking-[0.2em] mb-8 flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88]" />
                烹飪執行協議
            </h3>
            <div className="space-y-10">
                {steps.map((step, index) => (
                    <div key={index} className="relative pl-10">
                        <div className="absolute left-0 top-0 w-6 h-6 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-[#00ff88] text-[9px] font-black">
                            {index + 1}
                        </div>
                        {index < steps.length - 1 && (
                            <div className="absolute left-3 top-6 bottom-[-40px] w-[1px] bg-gradient-to-b from-[#00ff88]/20 to-transparent" />
                        )}
                        <h4 className="font-black text-xs text-white uppercase tracking-widest mb-2">{step.title}</h4>
                        <p className="text-[11px] text-gray-500 font-medium leading-relaxed uppercase tracking-tighter">{step.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}
