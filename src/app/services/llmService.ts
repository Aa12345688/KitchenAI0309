/**
 * LLM 食譜生成服務 (LLM Recipe Generation Service)
 * 
 * 這個模組負責與大語言模型 (例如 GPT-4, Gemini) 進行通訊，
 * 根據用戶提供的食材清單動態生成食譜建議。
 */

export interface LLMRecipeRequest {
    ingredients: string[]; // 用戶擁有的食材
    preferences?: string; // 用戶偏好 (可選)
}

export interface LLMRecipe {
    id: string;
    name: string;
    image: string;
    time: string;
    difficulty: "easy" | "medium" | "hard";
    category: "vegetable" | "fruit" | "meat" | "mixed";
    requiredIngredients: string[];
    description: string;
    matchScore: number;
}

// 獲取 API KEY 核心邏輯 (Get API KEY Logic)
// 優先從 Vite 環境變數中讀取 (VITE_LLM_API_KEY)
// 這能確保金鑰不會直接洩露在前端代碼中，並支持不同環境的靈活配置。
export const getApiKey = () => {
    return import.meta.env.VITE_LLM_API_KEY || "";
};

class LLMService {
    /**
     * 調用後端 API 生成食譜 (Forward to backend API)
     * @param request 包含食材清單的請求對象
     */
    async generateRecipes(request: LLMRecipeRequest): Promise<LLMRecipe[]> {
        const ingredients = request.ingredients;
        console.log("正在發送食譜生成請求至後端...", ingredients);

        try {
            const apiUrl = import.meta.env.VITE_DETECTION_API_URL || "http://localhost:8000";
            const response = await fetch(`${apiUrl}/api/generate-recipe`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "bypass-tunnel-reminder": "true"
                },
                body: JSON.stringify({
                    selected_ingredients: ingredients,
                    preferences: request.preferences
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `伺服器回應錯誤: ${response.status}`);
            }

            const recipes = await response.json();

            // 轉換數據格式以匹配前端需求 (Ensure camelCase fields)
            return recipes.map((r: any) => ({
                id: r.id || `recipe-${Date.now()}-${Math.random()}`,
                name: r.name || "AI 創意料理",
                image: r.image || "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=800",
                time: r.time || "25 MIN",
                difficulty: r.difficulty?.toLowerCase() || "medium",
                category: r.category || "mixed",
                requiredIngredients: r.requiredIngredients || ingredients,
                description: r.description || "智慧生成的精選食譜。",
                matchScore: r.matchScore || 85
            }));

        } catch (error) {
            console.error("LLMService 調用失敗:", error);
            // 本地保底邏輯 (Local fallback in case of absolute failure)
            return [
                {
                    id: "local-fallback",
                    name: "全能型食材濃湯 (離線)",
                    image: "https://images.unsplash.com/photo-1547592166-23ac45744acd?q=80&w=800",
                    time: "25 min",
                    difficulty: "medium",
                    category: "mixed",
                    requiredIngredients: ingredients.slice(0, 3),
                    description: "當網路連線不穩時提供的基礎應急食譜。",
                    matchScore: 85
                }
            ];
        }
    }
}

export const llmService = new LLMService();
