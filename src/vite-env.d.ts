/**
 * Vite 環境變數型別定義 (Vite Environment Variables Types)
 * 
 * 這裡定義了應用程式中使用的環境變量結構，
 * 確保在開發過程中可以使用 import.meta.env.VITE_XXX 並獲得 TypeScript 支援。
 */
/// <reference types="vite/client" />

interface ImportMetaEnv {
    // LLM 服務所需的 API 金鑰 (Gemini 或 OpenAI)
    readonly VITE_LLM_API_KEY: string
    readonly VITE_DETECTION_API_URL: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
