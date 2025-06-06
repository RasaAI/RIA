import { Routes } from '@angular/router';

export const appRoutes: Routes = [
  {
    path: 'Liquidity-risk',
    loadChildren: () =>
      import('./pages/Liquidity-risk/Liquidity-risk.module').then(m => m.MarketRiskModule)
      // Make sure the path points to the correct file and the class exported is MarketRiskModule
  },
  {
    path: '',
    redirectTo: '/market-risk',
    pathMatch: 'full'
  },
  {
    path: '**',
    redirectTo: '/mar'
  }
];
