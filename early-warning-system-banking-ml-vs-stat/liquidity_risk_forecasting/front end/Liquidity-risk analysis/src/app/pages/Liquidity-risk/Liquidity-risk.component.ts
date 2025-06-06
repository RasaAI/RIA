import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

interface RiskData {
  predict: string;
  reasons: string[];
}

@Component({
  selector: 'app-market-risk',
  imports: [CommonModule, FormsModule],
  templateUrl: './Liquidity-risk.component.html',
  styleUrls: ['./Liquidity-risk.component.css'],
})
export class MarketRiskComponent {
  selectedOption: string = 'import';
  showResult: boolean = false;

  constructor(private http: HttpClient) {}

  selectedFile: File | null = null;
  csvData: string[][] = [];

  csvFiles: { file: File | null; weightage: number | null; csvData: string[][] }[] = [
    { file: null, weightage: null, csvData: [] },
  ];

  // Tab switch: reset result and data
  setSelectedOption(option: string): void {
    this.selectedOption = option;
    this.clearResult();
  }

  // Handle single stock file upload
  onFileChanges(event: any): void {
    const file = event.target.files[0];
    this.selectedFile = file;
    console.log('File selected:', this.selectedFile);
  }

  // Handle portfolio file upload
  onFileChange(event: any, index: number): void {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const parsed = this.parseCSV(text);
      this.csvFiles[index].file = file;
      this.csvFiles[index].csvData = parsed;
    };
    reader.readAsText(file);
  }

  // CSV parser
  parseCSV(text: string): string[][] {
    return text
      .trim()
      .split('\n')
      .map((row) => row.split(',').map((cell) => cell.trim()));
  }

  // Add row if last is complete
  addCsvFile(): void {
    const last = this.csvFiles[this.csvFiles.length - 1];
    if (last.file && last.weightage !== null) {
      this.csvFiles.push({ file: null, weightage: null, csvData: [] });
    } else {
      alert('Please upload a file and enter weightage before adding a new row.');
    }
  }

  // Remove a file row (except first)
  removeCsvFile(index: number): void {
    if (index > 0) {
      this.csvFiles.splice(index, 1);
    }
  }

  // Submit portfolio
  submitPortfolio(): void {
  const totalWeight = this.csvFiles.reduce((sum, f) => sum + (f.weightage || 0), 0);
  if (totalWeight !== 100) {
    alert(`Total weightage must be exactly 100%. Current total: ${totalWeight}%`);
    return;
  }

  // Check all files and weightages present
  const incomplete = this.csvFiles.some(f => !f.file || f.weightage === null);
  if (incomplete) {
    alert('Please complete all file uploads and weightage values.');
    return;
  }

  // Prepare FormData
  const formData = new FormData();

  // Append files and weightages
  this.csvFiles.forEach((fileObj, index) => {
    if (fileObj.file) {
      formData.append(`file${index}`, fileObj.file, fileObj.file.name);
      formData.append(`weightage${index}`, fileObj.weightage!.toString());
    }
  });

  // Optionally, send total files count or other metadata
  formData.append('fileCount', this.csvFiles.length.toString());

  // Send to your API
  this.http.post<any>('http://127.0.0.1:8000/upload-portfolio', formData).subscribe({
    next: (res: any) => {
      console.log('Portfolio upload successful:', res);
      alert('Portfolio submitted successfully!');
      this.showResult = true;
    },
    error: (err: any) => {
      console.error('Portfolio upload error:', err);
      alert('Failed to submit portfolio.');
            this.showResult = true;

    },
  });
}


  // Handle single stock result generation
  toggleResult(): void {
    if (!this.selectedFile) {
      alert('Please select a CSV file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post<any>('http://127.0.0.1:8000/upload-portfolio', formData).subscribe({
      next: (res: any) => {
        console.log('Upload results:', res);
        this.riskData = res;
        this.showResult = true;
      },
      error: (err: any) => {
        console.error('Upload error:', err);
        alert('Upload failed.');
            this.showResult = true;

      },
    });
  }

 riskData!: RiskData;
  // showResult = false;
  // private apiUrl = 'http://127.0.0.1:8000/upload-portfolio'; // Replace with your actual API URL

  // constructor(private http: HttpClient) {}

  // ngOnInit() {
  //   this.getRiskData().subscribe({
  //     next: (data) => {
  //       this.riskData = data;
  //       this.showResult = true;
  //     },
  //     error: (error) => {
  //       console.error('Error fetching risk data:', error);
  //     }
  //   });
  // }

  // Function to call API and return Observable<RiskData>
  // getRiskData(): Observable<RiskData> {
  //   return this.http.get<RiskData>(this.apiUrl);
  // }

  // Positions for reasons UI
  getReasonPosition(index: number): string {
    const positions = [
      'top-0 left-1/2 transform -translate-x-1/2 -translate-y-full',   // Top
      'right-0 top-1/2 transform translate-x-full -translate-y-1/2',   // Right
      'bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full', // Bottom
      'left-0 top-1/2 transform -translate-x-full -translate-y-1/2'    // Left
    ];
    return positions[index] || 'top-1/2 left-1/2';
  }

  // Clear result and reset data
  clearResult(): void {
    this.showResult = false;
    this.selectedFile = null;
    this.csvData = [];

    this.csvFiles = [
      { file: null, weightage: null, csvData: [] }
    ];
  }
}


