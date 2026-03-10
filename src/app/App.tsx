import { RouterProvider } from 'react-router';
import { router } from './routes';
import { IngredientProvider } from './services/IngredientContext';

// 應用的根組件，負責引導路由配置
export default function App() {
  return (
    <IngredientProvider>
      <RouterProvider router={router} />
    </IngredientProvider>
  );
}
