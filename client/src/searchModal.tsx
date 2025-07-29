import React, { useState } from "react";

interface MatchResult {
  filename: string;
  score: GLfloat;
  snippet: string;
}

export default function SearchModal({ onClose }: { onClose: () => void }) {
  const [keyword, setKeyword] = useState("");
  const [results, setResults] = useState<MatchResult[]>([]);
  const [loading, setLoading] = useState(false);

  const searchKeyword = async () => {
    if (!keyword.trim()) return;
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/semantic-search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: keyword })
      });

      const data = await res.json();
      setResults(data.matches || []);
    } catch (err) {
      console.error("Search failed:", err);
      setResults([]);
    }

    setLoading(false);
  };

const getSharePointLink = (filename: string) => {
  // Make sure the SharePoint base URL is correct for your case
  const sharePointBaseUrl = "https://inovarcloud-my.sharepoint.us/personal/lam_nguyenngoc_spartronics_com/_layouts/15/onedrive.aspx";

  // Define the folder path on SharePoint where documents are located
  const folderPath = "Documents/Dylan%20Project"; // Adjust this if needed

  // Encode the filename to ensure it is URL-safe
  const encodedFilename = encodeURIComponent(filename);

  // Construct the complete SharePoint URL pointing to the document
  const filePath = `%2Fpersonal%2Flam%5Fnguyenngoc%5Fspartronics%5Fcom%2F${folderPath}%2F${encodedFilename}`;
  const parentPath = `%2Fpersonal%2Flam%5Fnguyenngoc%5Fspartronics%5Fcom%2F${folderPath}`;

  const link = `${sharePointBaseUrl}?id=${filePath}&parent=${parentPath}`;

  return link;
};


  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white w-full max-w-xl p-6 rounded-lg shadow-lg relative">
        <button
          onClick={onClose}
          className="absolute top-3 right-4 text-gray-400 hover:text-black text-2xl"
        >
          ×
        </button>

        <h3 className="text-lg font-semibold mb-4">Search PDFs by Keyword</h3>

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

        <div className="mt-4 max-h-60 overflow-y-auto">
          {loading ? (
            <p className="text-gray-600">Searching...</p>
          ) : results.length > 0 ? (
            results.map((r, idx) => (
              <div key={idx} className="border-t pt-2 mt-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <a
                    href={getSharePointLink(r.filename)}  // Add the SharePoint link here
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-semibold text-black-600 hover:text-blue-600"
                  >
                    {r.filename}
                  </a>
                  <p className="font-semibold text-sm text-gray-600">Score: {(r.score * 100).toFixed(1)}%</p>
                  <p className="text-sm text-gray-700 flex-1 truncate">{r.snippet}</p>
                </div>
              </div>
            ))
          ) : (
            <p className="text-gray-500 mt-4">No results yet. Try a search!</p>
          )}
        </div>
      </div>
    </div>
  );
}
