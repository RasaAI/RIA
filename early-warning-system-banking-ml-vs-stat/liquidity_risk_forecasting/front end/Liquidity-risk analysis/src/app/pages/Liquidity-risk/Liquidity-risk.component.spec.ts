import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MarketRiskComponent } from './Liquidity-risk.component';

describe('MarketRiskComponent', () => {
  let component: MarketRiskComponent;
  let fixture: ComponentFixture<MarketRiskComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MarketRiskComponent]  // ✅ only if it's a standalone component
    })
    .compileComponents();

    fixture = TestBed.createComponent(MarketRiskComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
