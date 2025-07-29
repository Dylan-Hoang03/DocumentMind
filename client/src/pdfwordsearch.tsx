import React, { useState } from "react";
import { Search } from "lucide-react"; // Make sure this is installed

const API = "http://localhost:5000"; // Adjust to match your backend

interface MatchResult {
  filename: string;
  snippet: string;
}

export default function PDFKeywordSearch() {
  const [showModal, setShowModal] = useState(false);
  const [keyword, setKeyword] = useState("");
  const [results, setResults] = useState<MatchResult[]>([]);
  const [loading, setLoading] = useState(false);

  const searchKeyword = async () => {
    if (!keyword.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${API}/search-keyword?keyword=${encodeURIComponent(keyword)}`);
      const data = await res.json();
      setResults(data.matches || []);
    } catch (err) {
      console.error("Search failed:", err);
      setResults([]);
    }
    setLoading(false);
  };

  return (
    <div className="p-6">
      {/* Header row with Add PDF and Search Icon */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">PDFs</h2>

        <div className="flex items-center gap-2">
          {/* Add PDF Button */}
          <button className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
            Add PDF
          </button>

          {/* Search Icon Button */}
          <button
            onClick={() => setShowModal(true)}
            className="p-2 rounded hover:bg-gray-100 text-gray-600"
            aria-label="Search PDFs"
            title="Search PDFs"
          >
            <Search className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
          <div className="bg-white w-full max-w-xl p-6 rounded-lg shadow-lg relative">
            {/* Close button */}
            <button
              onClick={() => setShowModal(false)}
              className="absolute top-3 right-4 text-gray-400 hover:text-black text-2xl"
            >
              ×
            </button>

            {/* Title */}
            <h3 className="text-lg font-semibold mb-4">Search PDFs by Keyword</h3>

            {/* Input + Button */}
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 border border-gray-300 rounded px-3 py-2"
                placeholder="Type keyword (e.g., invoice, compliance)..."
                value={keyword}
                onChange={e => setKeyword(e.target.value)}
              />
              <button
                onClick={searchKeyword}
                className="bg-blue-600 text-white px-4 py-2 rounded"
              >
                Search
              </button>
            </div>

            {/* Results */}
            <div className="mt-4 max-h-60 overflow-y-auto">
              {loading ? (
                <p className="text-gray-600">Searching...</p>
              ) : results.length > 0 ? (
                results.map((r, idx) => (
                  <div key={idx} className="border-t pt-2 mt-2">
                    <p className="font-semibold">{r.filename}</p>
                    <p className="text-sm text-gray-700">{r.snippet}</p>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 mt-4">No results yet. Try a search!</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
